# This file is part of the scanning-squid package.
#
# Copyright (c) 2018 Logan Bishop-Van Horn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#: Various Python utilities
from typing import Dict, List, Sequence, Any, Union, Tuple
import time

#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType, FrequencyUnits, Level

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, MeasPlot
from .microscope import Microscope
import utils
import matplotlib.pyplot as plt

#: Pint for manipulating physical units
from pint import UnitRegistry
ureg = UnitRegistry()
#: Tell UnitRegistry instance what a Phi0 is, and that Ohm = ohm
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')
ureg.load_definitions('./squid_units.txt')

import logging
log = logging.getLogger(__name__)

class SusceptometerMicroscope(Microscope):
    """Scanning SQUID susceptometer microscope class.
    """
    def __init__(self, config_file: str, temp: str, ureg: Any=ureg, log_level: Any=logging.INFO,
                 log_name: str=None, **kwargs) -> None:
        """
        Args:
            config_file: Path to microscope configuration JSON file.
            temp: 'LT' or 'RT', depending on whether the microscope is cold or not.
                Sets the voltage limits for the scanner and Attocubes.
            ureg: pint UnitRegistry for managing physical units.
            log_level: e.g. logging.DEBUG or logging.INFO
            log_name: Log file will be saved as logs/{log_name}.log.
                Default is the name of the microscope configuration file.
            **kwargs: Keyword arguments to be passed to Station constructor.
        """
        super().__init__(config_file, temp, ureg, log_level, log_name, **kwargs)

    def get_prefactors(self, measurement: Dict[str, Any], update: bool=True) -> Dict[str, Any]:
        """For each channel, calculate prefactors to convert DAQ voltage into real units.
        Args:
            measurement: Dict of measurement parameters as defined
                in measurement configuration file.
            update: Whether to query instrument parameters or simply trust the
                latest values (should this even be an option)?
        Returns:
            Dict[str, pint.Quantity]: prefactors
                Dict of {channel_name: prefactor} where prefactor is a pint Quantity.
        """
        mod_width = self.Q_(self.SQUID.metadata['modulation_width'])
        prefactors = {}
        for ch in measurement['channels']:
            prefactor = 1
            if ch == 'MAG':
                prefactor /= mod_width
            elif ch in ['SUSCX', 'SUSCY']:
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                susc_sensitivity = snap['sensitivity']['value']
                amp = snap['amplitude']['value'] * self.ureg(snap['amplitude']['unit'])
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor *=  (r_lead / amp) / (mod_width * 10 / susc_sensitivity)
            elif ch == 'CAP':
                snap = getattr(self, 'CAP_lockin').snapshot(update=update)['parameters']
                cap_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor /= (self.Q_(self.scanner.metadata['cantilever']['calibration']) * 10 / cap_sensitivity)
            elif ch == 'LIA':
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                prefactor /= r_lead
            elif ch == 'IV':
                snap = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                IV_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor *= IV_sensitivity/10
            prefactor = self.ureg.Quantity(str(prefactor))
            prefactor /= measurement['channels'][ch]['gain']
            prefactors.update({ch: prefactor.to('{}/V'.format(measurement['channels'][ch]['unit']))})
        return prefactors

    def scan_surface(self, scan_params: Dict[str, Any], ring: int=None) -> None:
        """
        Scan the current surface while acquiring data in the channels defined in
        measurement configuration file (e.g. MAG, SUSCX, SUSCY, CAP).
        Args:
            scan_params: Dict of scan parameters as defined
                in measuremnt configuration file.
        Returns:
            Tuple[qcodes.DataSet, plots.ScanPlot]: data, plot
                qcodes DataSet containing acquired arrays and metdata,
                and ScanPlot instance populated with acquired data.
        """
        if not self.atto.surface_is_current:
            raise RuntimeError('Surface is not current. Aborting scan.')
        surface_type = scan_params['surface_type'].lower()
        if surface_type not in ['plane', 'surface']:
            raise ValueError('surface_type must be "plane" or "surface".')

        old_pos = self.scanner.position()
        
        daq_config = self.config['instruments']['daq']
        ao_channels = daq_config['channels']['analog_outputs']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = scan_params['channels']
        channels = {}
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())

        daq_name = daq_config['name']
        #: DAQ AI sampling rate is divided amongst all active AI channels
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels
        
        fast_ax = scan_params['fast_ax'].lower()
        slow_ax = 'x' if fast_ax == 'y' else 'y'
        
        pix_per_line = scan_params['scan_size'][fast_ax]
        line_duration = pix_per_line * self.ureg('pixels') / self.Q_(scan_params['scan_rate'])
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        
        height = self.Q_(scan_params['height']).to('V').magnitude
        
        scan_vectors = utils.make_scan_vectors(scan_params, self.ureg)
        #scan_grids = utils.make_scan_grids(scan_vectors, slow_ax, fast_ax,
        #                                   pts_per_line, plane, height)
        plane = self.scanner.metadata['plane']
        if surface_type == 'plane':
            scan_grids = utils.make_scan_surface(surface_type, scan_vectors, slow_ax, fast_ax,
                                                pts_per_line, plane, height)
        else:
            scan_grids = utils.make_scan_surface(surface_type, scan_vectors, slow_ax, fast_ax,
                                                pts_per_line, plane, height, interpolator=self.scanner.surface_interp)
        utils.validate_scan_params(self.scanner.metadata, scan_params,
                                   scan_grids, self.temp, self.ureg, log)
        self.scanner.goto([scan_grids[axis][0][0] for axis in ['x', 'y', 'z']])
        self.set_lockins(scan_params)
        #: get channel prefactors in pint Quantity form
        prefactors = self.get_prefactors(scan_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = scan_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task = nidaqmx.Task('scan_plane_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            #self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,
                                      samples_to_read=pts_per_line, target_points=pix_per_line,
                                      #: Very important to synchronize AOs and AIs
                                      clock_src='ao/SampleClock')
        self.add_component(self.daq_ai)
        slow_ax_position = getattr(self.scanner, 'position_{}'.format(slow_ax))
        slow_ax_start = scan_vectors[slow_ax][0]
        slow_ax_end = scan_vectors[slow_ax][-1]
        slow_ax_step = scan_vectors[slow_ax][1] - scan_vectors[slow_ax][0]
        #: There is probably a counter built in to qc.Loop, but I couldn't find it
        loop_counter = utils.Counter()
        scan_plot = ScanPlot(scan_params, self.ureg)
        #: IPZ 07/08/19: If scanning in preparation for a ring measurement, stay at last pos.
        if ring:
            loop = qc.Loop(slow_ax_position.sweep(start=slow_ax_start,
                                              stop=slow_ax_end,
                                              step=slow_ax_step), delay=0.1
                ).each(
                #: Create AO task and queue data to be written to AOs
                qc.Task(self.scanner.scan_line, scan_grids, ao_channels, daq_rate, loop_counter),
                #: Start AI task; acquisition won't start until AO task is started
                qc.Task(ai_task.start),
                #: Start AO task
                qc.Task(self.scanner.control_ao_task, 'start'),
                #: Acquire voltage from all active AI channels
                self.daq_ai.voltage,
                qc.Task(ai_task.wait_until_done),
                qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
                qc.Task(ai_task.stop),
                #: Stop and close AO task so that AOs can be used for goto
                qc.Task(self.scanner.control_ao_task, 'stop'),
                qc.Task(self.scanner.control_ao_task, 'close'),
                qc.Task(self.scanner.goto_start_of_next_line, scan_grids, loop_counter),
                #: Update and save plot
                qc.Task(scan_plot.update, qc.loops.active_data_set, loop_counter),
                qc.Task(scan_plot.save),
                qc.Task(loop_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(self.daq_ai.close),
                #qc.Task(self.daq_ai.clear_instances),
                #qc.Task(self.scanner.goto, old_pos),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004)
            )
        else:
            loop = qc.Loop(slow_ax_position.sweep(start=slow_ax_start,
                                              stop=slow_ax_end,
                                              step=slow_ax_step), delay=0.1
                ).each(
                #: Create AO task and queue data to be written to AOs
                qc.Task(self.scanner.scan_line, scan_grids, ao_channels, daq_rate, loop_counter),
                #: Start AI task; acquisition won't start until AO task is started
                qc.Task(ai_task.start),
                #: Start AO task
                qc.Task(self.scanner.control_ao_task, 'start'),
                #: Acquire voltage from all active AI channels
                self.daq_ai.voltage,
                qc.Task(ai_task.wait_until_done),
                qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
                qc.Task(ai_task.stop),
                #: Stop and close AO task so that AOs can be used for goto
                qc.Task(self.scanner.control_ao_task, 'stop'),
                qc.Task(self.scanner.control_ao_task, 'close'),
                qc.Task(self.scanner.goto_start_of_next_line, scan_grids, loop_counter),
                #: Update and save plot
                qc.Task(scan_plot.update, qc.loops.active_data_set, loop_counter),
                qc.Task(scan_plot.save),
                qc.Task(loop_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(self.daq_ai.close),
                #qc.Task(self.daq_ai.clear_instances),
                qc.Task(self.scanner.goto, old_pos),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(scan_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=scan_params['fname'])
        #: Run the loop
        try:
            loop.run()
            log.info('Scan completed. DataSet saved to {}.'.format(data.location))
        #: If loop is aborted by user:
        except KeyboardInterrupt:
            log.warning('Scan interrupted by user. Going to [0, 0, -10] V.')
            #: Stop 'scan_plane_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #: If there's an active AO task, close it so that we can use goto
            try:
                self.scanner.control_ao_task('stop')
                self.scanner.control_ao_task('close')
            except:
                pass
            self.scanner.goto([0, 0, -10])
            #self.CAP_lockin.amplitude(0.004)
            #self.SUSC_lockin.amplitude(0.004)
            log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
        utils.scan_to_mat_file(data, real_units=True, interpolator=self.scanner.surface_interp)
        #return data, scan_plot

    def meas_ring(self, meas_params: Dict[str, Any]) -> None:
        """
        Acquires data in the channels defined in measurement configuration 
        file (e.g. MAG, LIA, CAP) and averages across lockin periods.
        Args:
            meas_params: Dict of measurement parameters as defined
                in measuremnt configuration file.
        """
        if not self.atto.surface_is_current:
            raise RuntimeError('Surface is not current. Aborting scan.')

        #old_pos = self.scanner.position() #: Don't need to explicitly recond scanner position because it will bein the snapshot file
        #: Set up lockin.
        response = input('Lockin set to EXTERNAL? y/[n] ')
        if response.lower() != 'y':
            log.warning('Please set lockin to EXTERNAL.')
            return
        self.set_lockins(meas_params)
        lia_freq = self.Q_(meas_params['channels']['LIA']['lockin']['frequency']).to('Hz').magnitude
        
        #: Get DAQ configuration and channel names for counter, pfi, and analog input channels.
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        co_channel = '{}/ctr{}'.format(daq_name, daq_config['channels']['counter_outputs']['lockin_trigger']) 
        pfi_channel = '/{}/PFI{}'.format(daq_name, daq_config['channels']['pf_inputs']['meas_trigger'])
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = meas_params['channels']
        channels = {}
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())
        #: DAQ AI sampling rate is divided amongst all active AI channels
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels

        #: Calculate measurement constants. Floor division ensures a whole number of scans per period.
        scans_per_period = int(daq_rate // lia_freq)
        co_freq = daq_rate / scans_per_period
        nscans = scans_per_period * int(meas_params['nperiods_per_block'])
        nblocks = int(meas_params['nblocks'])
        duration = nscans / daq_rate
        if duration > 25:
            raise ValueError('"nperiods_per_block" must be less than or equal to 25 s / "daq_rate".')

        #: get channel prefactors in pint Quantity form
        prefactors = self.get_prefactors(meas_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = meas_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})

        ai_task = nidaqmx.Task('meas_ring_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            #self.daq_ai.clear_instances()
            self.daq_ai.close()
        # Wait for all samples to be acquired before reading them in. Fold over periods and average.
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,  ring=1, trigger_src=pfi_channel,
                                     samples_to_read=nscans, target_points=scans_per_period)
        #self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task, ring=1, trigger_src=pfi_channel,
        #                              samples_to_read=nscans)
        self.add_component(self.daq_ai)
        #: Create counter output task that will trigger the lockin and ai_task.
        #: Shouldn't need explicit write command (see read/write test on github)
        co_task = nidaqmx.Task('meas_ring_co_task')
        co_task.co_channels.add_co_pulse_chan_freq(
            co_channel, daq_name, 
            units=FrequencyUnits.HZ, 
            idle_state=Level.LOW, 
            initial_delay=0.0, 
            freq=co_freq, 
            duty_cycle=0.5)
        co_task.timing.cfg_implicit_timing(sample_mode=AcquisitionType.CONTINUOUS)
            #: Try this if it doesn't work as is.
            #sample = CtrFreq(co_freq, dut_cycle=0.5) 
            #co_task.write(sample, auto_start=False) 
        #: Start the counter output
        co_task.start()
        #: Add dummy instrument to sweep over n blocks.
        if 'block' in self.components:
            self.remove_component('block')
        if hasattr(self, 'block'):
            self.block.close()
        self.block = utils.DummyBlock('block')
        self.add_component(self.block)
        block_counter = utils.Counter()
        meas_plot = MeasPlot(meas_params, duration, co_freq, nscans, scans_per_period, self.ureg) 
        loop = qc.Loop(self.block.current_block.sweep(start=0, stop=(nblocks-1), step=1)
            ).each(
                #: Start ai task, which will begin acquiring samples when triggered by the CO output.
                qc.Task(ai_task.start),
                #: Read in the samples measured in this block.
                self.daq_ai.voltage,
                qc.Task(ai_task.wait_until_done),
                qc.Task(ai_task.stop),
                #: Update and save plot
                qc.Task(meas_plot.update, qc.loops.active_data_set, block_counter),
                qc.Task(meas_plot.save),
                #: Advance to next block
                qc.Task(block_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                qc.Task(co_task.stop),
                qc.Task(co_task.close),
                qc.Task(self.daq_ai.close),
                qc.Task(self.block.close))
        loop.metadata.update(meas_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        #: Intialize the DataSet.
        data = loop.get_data_set(name=meas_params['fname'])
        #: Wait 5 s
        time.sleep(5)
        #: Run the loop
        try:
            loop.run()
            log.info('Measurement completed. DataSet saved to {}.'.format(data.location))
        #: If loop is aborted by user:
        except KeyboardInterrupt:
            log.warning('Measurement interrupted by user.')
            #: Stop ai and co tasks.
            ai_task.stop()
            ai_task.close()
            co_task.stop()
            co_task.close()
            log.info('Measurement aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
        self.remove_component('block')
        #: Save to .mat file
        utils.meas_to_mat_file(data, co_freq=co_freq, scans_per_period=scans_per_period, 
            real_units=True)

    def plot_T_vs_time(self):
        time_vec=[]
        sample_T=[]
        elapsed_time=0
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlabel('t(sec)')
        self.ax.set_ylabel('T(K)')
         
        while (True):
            sample_T.append(self.ls340.A.temperature())
            time_vec.append(elapsed_time)
            time.sleep(1)
            elapsed_time=elapsed_time+1
            self.ax.plot(time_vec,sample_T)
            self.fig.canvas.draw()


    """def fft_noise_plane(self, scan_params: Dict[str, Any]) -> None:
    """

    """
        Scan the current surface while acquiring fft data in the channels defined in
        measurement configuration file (e.g. MAG, SUSCX, SUSCY, CAP).
        Args:
            scan_params: Dict of scan parameters as defined
                in measuremnt configuration file.
        Returns:
            Tuple[qcodes.DataSet, plots.ScanPlot]: data, plot
                qcodes DataSet containing acquired arrays and metdata,
                and ScanPlot instance populated with acquired data.
    """
    """
        if not self.atto.surface_is_current:
            raise RuntimeError('Surface is not current. Aborting scan.')
        surface_type = scan_params['surface_type'].lower()
        if surface_type not in ['plane', 'surface']:
            raise ValueError('surface_type must be "plane" or "surface".')

        old_pos = self.scanner.position()
        
        daq_config = self.config['instruments']['daq']
        ao_channels = daq_config['channels']['analog_outputs']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = scan_params['channels']
        channels = {}
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())

        daq_name = daq_config['name']
        #: DAQ AI sampling rate is divided amongst all active AI channels
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels
        
        fast_ax = scan_params['fast_ax'].lower()
        slow_ax = 'x' if fast_ax == 'y' else 'y'
        
        pix_per_line = scan_params['scan_size'][fast_ax]
        line_duration = pix_per_line * self.ureg('pixels') / self.Q_(scan_params['scan_rate'])
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        
        height = self.Q_(scan_params['height']).to('V').magnitude
        
        scan_vectors = utils.make_scan_vectors(scan_params, self.ureg)
        #scan_grids = utils.make_scan_grids(scan_vectors, slow_ax, fast_ax,
        #                                   pts_per_line, plane, height)
        plane = self.scanner.metadata['plane']
        if surface_type == 'plane':
            scan_grids = utils.make_scan_surface(surface_type, scan_vectors, slow_ax, fast_ax,
                                                pts_per_line, plane, height)
        else:
            scan_grids = utils.make_scan_surface(surface_type, scan_vectors, slow_ax, fast_ax,
                                                pts_per_line, plane, height, interpolator=self.scanner.surface_interp)
        utils.validate_scan_params(self.scanner.metadata, scan_params,
                                   scan_grids, self.temp, self.ureg, log)
        self.scanner.goto([scan_grids[axis][0][0] for axis in ['x', 'y', 'z']])
        self.set_lockins(scan_params)
        #: get channel prefactors in pint Quantity form
        prefactors = self.get_prefactors(scan_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = scan_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        samplerate = scan_params['samplerate']
        sampleduration = scan_params['sampleduration']
        navg = scan_params['navg']
        fmax = scan_params['fmax']
        nsamples = int(samplerate * sampleduration)
        v_fft_avg = np.zeros((nsamples // 2,))
        ai_task = nidaqmx.Task('fft_noise_plane_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            #self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, samplerate, channels, ai_task,
                                        samples_to_read=nsamples, timeout=sampleduration+10)
        self.add_component(self.daq_ai)
       
        
        mdict.update({
            'v_fft_avg': v_fft_avg[freqs < fmax],
            'sig_fft_avg': sig_fft_avg[freqs < fmax],
            'freqs': freqs[freqs < fmax]})
        
        slow_ax_position = getattr(self.scanner, 'position_{}'.format(slow_ax))
        slow_ax_start = scan_vectors[slow_ax][0]
        slow_ax_end = scan_vectors[slow_ax][-1]
        slow_ax_step = scan_vectors[slow_ax][1] - scan_vectors[slow_ax][0]
        #: There is probably a counter built in to qc.Loop, but I couldn't find it
        loop_counter = utils.Counter()
        scan_plot = ScanPlot(scan_params, self.ureg)
        
        loop = qc.Loop(slow_ax_position.sweep(start=slow_ax_start,
                                          stop=slow_ax_end,
                                          step=slow_ax_step), delay=0.1
            ).each(
            #: Create AO task and queue data to be written to AOs
            qc.Task(self.scanner.scan_line, scan_grids, ao_channels, daq_rate, loop_counter),
            #: Start AI task; acquisition won't start until AO task is started
            qc.Task(ai_task.start),
            #: Start AO task
            qc.Task(self.scanner.control_ao_task, 'start'),
            #: Acquire voltage from all active AI channels for navg samples
            v_fft_avg = np.zeros((nsamples // 2,)),
            for i in range(navg):
                data_v = self.daq_ai.voltage()[0].T
                Fs = nsamples / sampleduration
                v_fft = np.fft.fft(data_v) / (nsamples / np.sqrt(2 * sampleduration))
                v_fft_abs = np.abs(v_fft[:nsamples//2])
                freqs = np.fft.fftfreq(nsamples, d=1/Fs)[:nsamples//2]
                v_fft_avg += v_fft_abs,
            v_fft_avg = v_fft_avg / navg,
            sig_fft_avg = prefactor.magnitude * v_fft_avg,
            qc.Task(ai_task.wait_until_done),
            qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
            qc.Task(ai_task.stop),
            #: Stop and close AO task so that AOs can be used for goto
            qc.Task(self.scanner.control_ao_task, 'stop'),
            qc.Task(self.scanner.control_ao_task, 'close'),
            qc.Task(self.scanner.goto_start_of_next_line, scan_grids, loop_counter),
            #: Update and save plot
            qc.Task(scan_plot.update, qc.loops.active_data_set, loop_counter),
            qc.Task(scan_plot.save),
            qc.Task(loop_counter.advance)
        ).then(
            qc.Task(ai_task.stop),
            qc.Task(ai_task.close),
            qc.Task(self.daq_ai.close),
            #qc.Task(self.daq_ai.clear_instances),
            qc.Task(self.scanner.goto, old_pos),
            #qc.Task(self.CAP_lockin.amplitude, 0.004),
            #qc.Task(self.SUSC_lockin.amplitude, 0.004)
        )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(scan_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=scan_params['fname'])
        #: Run the loop
        try:
            loop.run()
            log.info('Scan completed. DataSet saved to {}.'.format(data.location))
        #: If loop is aborted by user:
        except KeyboardInterrupt:
            log.warning('Scan interrupted by user. Going to [0, 0, -10] V.')
            #: Stop 'scan_plane_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #: If there's an active AO task, close it so that we can use goto
            try:
                self.scanner.control_ao_task('stop')
                self.scanner.control_ao_task('close')
            except:
                pass
            self.scanner.goto([0, 0, -10])
            #self.CAP_lockin.amplitude(0.004)
            #self.SUSC_lockin.amplitude(0.004)
            log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
        utils.scan_to_mat_file(data, real_units=True, interpolator=self.scanner.surface_interp)


       fft_noise(dev_name, channel, unit, prefactor, samplerate, sampleduration, navg, fmax):
    loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
    loc = loc_provider(DiskIO('.'), record={'name': 'fft_noise'})
    pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
    prefactor_str = {}
    prefactor.ito('{}/V'.format(unit))
    prefactor_str.update({list(channel.keys())[0]: '{} {}'.format(prefactor.magnitude, prefactor.units)})
    mdict = {
        'metadata': {
            #'channel': channel,
            #'unit': unit,
            #'prefactor': prefactor_str,
            'samplerate': samplerate,
            'sampleduration': sampleduration,
            'navg': navg,
            'fmax': fmax ,
            'location': loc
        }
    }
    nsamples = int(samplerate * sampleduration)
    v_fft_avg = np.zeros((nsamples // 2,))
    with nidaqmx.Task('fft_noise_ai_task') as ai_task:
        for inst in DAQAnalogInputs.instances():
            inst.close()
        daq_ai = DAQAnalogInputs('daq_ai', dev_name, samplerate, channel, ai_task,
            samples_to_read=nsamples, timeout=sampleduration+10)
        for i in range(navg):
            data_v = daq_ai.voltage()[0].T
            Fs = nsamples / sampleduration
            v_fft = np.fft.fft(data_v) / (nsamples / np.sqrt(2 * sampleduration))
            v_fft_abs = np.abs(v_fft[:nsamples//2])
            freqs = np.fft.fftfreq(nsamples, d=1/Fs)[:nsamples//2]
            v_fft_avg += v_fft_abs
        daq_ai.close()
        v_fft_avg = v_fft_avg / navg
        sig_fft_avg = prefactor.magnitude * v_fft_avg
        mdict.update({
            'v_fft_avg': v_fft_avg[freqs < fmax],
            'sig_fft_avg': sig_fft_avg[freqs < fmax],
            'freqs': freqs[freqs < fmax]})
        fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
        ax[0].loglog(freqs, v_fft_avg, lw=1)
        ax[0].set_ylabel('V/sqrt(Hz)')
        ax[1].loglog(freqs, sig_fft_avg, lw=1)
        ax[1].set_ylabel('{}/sqrt(Hz)'.format(unit))
        fig.suptitle(loc, x=0.5, y=1)
        for i in [0,1]:
            ax[i].set_xlabel('Frequency [Hz]')
            ax[i].grid()
        plt.savefig(loc + '/fft_noise.png')
        io.savemat(loc + '/fft_noise.mat', mdict)
        return mdict
"""