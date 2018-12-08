"""
This file is part of the scanning-squid package.

Copyright (c) 2018 Logan Bishop-Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

#: Various Python utilities
from typing import Dict, List, Sequence, Any, Union, Tuple
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from scipy import io
import pathlib

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcodes.data.io import DiskIO

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from instruments.dg645 import DG645
from instruments.afg3000 import AFG3000
from plots import ScanPlot
from .microscope import Microscope
import utils

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

class SamplerMicroscope(Microscope):
    """Scanning SQUID sampler microscope class.
    """
    def __init__(self, config_file: str, temp: str, ureg: Any=ureg, log_level: Any=logging.INFO,
                 log_name: str=None, **kwargs) -> None:
        super().__init__(config_file, temp, ureg=ureg, log_level=log_level,
            log_name=log_name, **kwargs)
        self._add_delay_generator()
        self._add_function_generators()

    def _add_delay_generator(self):
        """Add SRS DG645 digital delay generator to SamplerMicroscope.
        """
        info = self.config['instruments']['delay_generator']
        name = 'dg'
        if hasattr(self, name):
            getattr(self, name, 'clear_instances')()
        self.remove_component(name)
        instr = DG645(name, info['address'], metadata=info)
        setattr(self, name, instr)
        self.add_component(getattr(self, name))
        log.info('{} successfully added to microscope.'.format(name))

    def _add_function_generators(self):
        """Add Tektronix AFG3000 series function generators to SamplerMicroscope.
        """
        cfg = self.config['instruments']['function_generators']
        for name, info in cfg.items():
            if hasattr(self, name):
                getattr(self, name).close()
            self.remove_component(name)
            instr = AFG3000(name, info['address'], info['channels'], metadata=info)
            setattr(self, name, instr)
            self.add_component(getattr(self, name))
            log.info('{} successfully added to microscope.'.format(name))

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
        prefactors = {}
        for ch in measurement['channels']:
            prefactor = self.Q_(1)
            if ch in ['v_comp', 'nothing']:
                snap = getattr(self, 'MAG_lockin').snapshot(update=update)['parameters']
                mag_sensitivity = snap['sensitivity']['value']
                #amp = snap['amplitude']['value'] * self.ureg(snap['amplitude']['unit'])
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor /=  self.Q_(10 / mag_sensitivity)
            elif ch == 'CAP':
                snap = getattr(self, 'CAP_lockin').snapshot(update=update)['parameters']
                cap_sensitivity = snap['sensitivity']['value']
                #: The factor of 10 here is because SR830 output gain is 10/sensitivity
                prefactor /= (self.Q_(self.scanner.metadata['cantilever']['calibration']) * 10 / cap_sensitivity)
            prefactor /= measurement['channels'][ch]['gain']
            prefactors.update({ch: prefactor})
        return prefactors

    def iv_mod_tek(self, ivm_params: Dict[str, Any]) -> Tuple[Dict[str, Any]]:
        """Measures IV characteristic at different mod coil voltages.

        Args:
            ivm_params: Dict of measurement parameters as definted in config_measurements json file.

        Returns:
            Tuple[Dict]: data_dict, metadict
                Dictionaries containing data arrays and instrument metadata.
        """

        data_dict = {}
        meta_dict = {}
        mod_range = [self.Q_(value).to('V').magnitude for value in ivm_params['mod_range']]
        bias_range = [self.Q_(value).to('V').magnitude for value in ivm_params['bias_range']]
        mod_vec = np.linspace(mod_range[0], mod_range[1], ivm_params['ntek2'])
        bias_vec = np.linspace(bias_range[0], bias_range[1], ivm_params['ntek1'])
        mod_grid, bias_grid  = np.meshgrid(mod_vec, bias_vec)
        data_dict.update({
            'mod_vec': {'array': mod_vec, 'unit': 'V'},
            'bias_vec': {'array': bias_vec, 'unit': 'V'},
            'mod_grid': {'array': mod_grid, 'unit': 'V'},
            'bias_grid': {'array': bias_grid, 'unit': 'V'}
            })
        ivmX = np.full_like(mod_grid, np.nan, dtype=np.double)
        ivmY = np.full_like(mod_grid, np.nan, dtype=np.double)
        
        #: Set AFG output channels
        self.afg1.voltage_low1('0V')
        self.afg1.voltage_high1('{}V'.format(bias_range[0]))

        self.afg1.voltage_low2('0V')
        self.afg1.voltage_high2('{}V'.format(mod_range[0]))
        self.afg1.voltage_offset2('0V')

        #: Set pulse parameters
        for ch in [1, 2]:
            p = ivm_params['afg']['ch{}'.format(ch)]
            getattr(self.afg1, 'pulse_period{}'.format(ch))('{}us'.format(self.Q_(p['period']).to('us').magnitude))
            getattr(self.afg1, 'pulse_width{}'.format(ch))('{}us'.format(self.Q_(p['width']).to('us').magnitude))
            getattr(self.afg1, 'pulse_trans_lead{}'.format(ch))('{}us'.format(self.Q_(p['lead']).to('us').magnitude))
            getattr(self.afg1, 'pulse_trans_trail{}'.format(ch))('{}us'.format(self.Q_(p['trail']).to('us').magnitude))
            getattr(self.afg1, 'pulse_delay{}'.format(ch))('{}us'.format(self.Q_(p['delay']).to('us').magnitude))

        #: Get instrument metadata and prefactors
        lockin_snap = self.MAG_lockin.snapshot(update=True)
        lockin_meta = {}
        for param in ['time_constant', 'sensitivity', 'phase', 'reserve', 'filter_slope']:
            lockin_meta.update({param: lockin_snap['parameters'][param]})
        meta_dict.update({'metadata':
                            {'lockin': lockin_meta,
                             'afg': self.afg1.snapshot(update=True),
                             'ivm_params': ivm_params}
                        })
        prefactor = 1 / (10 / lockin_snap['parameters']['sensitivity']['value'])
        prefactor /= ivm_params['channels']['lockinX']['gain']
        delay = ivm_params['delay_factor'] * lockin_snap['parameters']['time_constant']['value']

        fig, ax = plt.subplots(1, figsize=(4,3))
        ax.set_xlim(min(bias_range), max(bias_range))
        ax.set_xlabel('Bias [V]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title(ivm_params['channels']['lockinX']['label'])
        log.info('Starting iv_mod_tek.')
        try:
            for j in range(len(mod_vec)):
                dataX, dataY, dataX_avg, dataY_avg = (np.zeros(len(mod_vec)), ) * 4
                self.afg1.voltage_offset2('{}V'.format(mod_vec[j]))
                for _ in range(ivm_params['navg']):
                    for i in range(len(bias_vec)):
                        self.afg1.voltage_high1('{}V'.format(bias_vec[i]))
                        time.sleep(delay)
                        dataX[i] = self.MAG_lockin.X()
                        dataY[i] = self.MAG_lockin.Y()
                    dataX_avg += dataX
                    dataY_avg += dataY
                dataX_avg /= ivm_params['navg']
                dataY_avg /= ivm_params['navg']
                ivmX[:,j] = prefactor * dataX_avg
                ivmY[:,j] = prefactor * dataY_avg
                utils.clear_artists(ax)
                ax.plot(bias_vec, prefactor * dataX_avg, 'bo-')
                plt.tight_layout()
                fig.canvas.draw()
            fig.show()

        except KeyboardInterrupt:
            log.warning('Measurement aborted by user.')

        figX = plt.figure(figsize=(4,3))    
        plt.pcolormesh(mod_grid, bias_grid, ivmX)
        plt.xlabel('Modulation [V]')
        plt.ylabel('Bias [V]')
        plt.title(ivm_params['channels']['lockinX']['label'])
        figX.tight_layout(rect=[0, 0.03, 1, 0.95])
        cbarX = plt.colorbar()
        cbarX.set_label('Voltage [V]')

        figY = plt.figure(figsize=(4,3))    
        plt.pcolormesh(mod_grid, bias_grid, ivmY)
        plt.xlabel('Modulation [V]')
        plt.ylabel('Bias [V]')
        plt.title(ivm_params['channels']['lockinY']['label'])
        figY.tight_layout(rect=[0, 0.03, 1, 0.95])
        cbarY = plt.colorbar()
        cbarY.set_label('Voltage [V]')

        data_dict.update({
            'lockinX': {'array': ivmX, 'unit': 'V'},
             'lockinY': {'array': ivmY, 'unit': 'V'}
            })

        if ivm_params['save']:
            #: Get/create data location
            loc_provider = qc.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
            loc = loc_provider(DiskIO('.'), record={'name': ivm_params['fname']})
            pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
            #: Save arrays to mat
            io.savemat('{}/{}'.format(loc, ivm_params['fname']), data_dict)
            #: Save metadata to json
            with open(loc + '/metadata.json', 'w') as f:
                try:
                    json.dump(meta_dict, f, sort_keys=True, indent=4, skipkeys=True)
                except TypeError:
                    pass
            #: Save figures to png
            figX.suptitle(loc)
            figX.savefig('{}/{}X.png'.format(loc, ivm_params['fname']))
            figY.suptitle(loc)
            figY.savefig('{}/{}Y.png'.format(loc, ivm_params['fname']))
            log.info('Data saved to {}.'.format(loc))

        return data_dict, meta_dict

    def iv_tek_mod_daq(self, ivm_params: Dict[str, Any]) -> None:
        """Performs digital feedback on mod coil to measure flux vs. delay.

        AFG ch1 is used for pulse generator bias.
        AFG ch2 is used for comparator bias.
        DG ch1 is used for pulse generator trigger.

        Args:
            ivm_params: Dict of measurement parameters as definted in config_measurements json file.

        Returns:
            Tuple[Dict]: data_dict, metadict
                Dictionaries containing data arrays and instrument metadata.
        """

        data_dict = {}
        meta_dict = {}
        daq_config = self.config['instruments']['daq']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = ivm_params['channels']
        channels = {}
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})

        vmod = self.Q_(ivm_params['vmod_initial']).to('V').magnitude
        vcomp_set = self.Q_(ivm_params['vcomp_set']).to('V').magnitude
        vmod_low = self.Q_(ivm_params['vmod_low']).to('V').magnitude
        vmod_high = self.Q_(ivm_params['vmod_high']).to('V').magnitude

        tsettle = self.Q_(ivm_params['tsettle']).to('s').magnitude
        tavg = self.Q_(ivm_params['tavg']).to('s').magnitude
        time_constant = self.Q_(ivm_params['time_constant'])
        
        period = self.Q_(ivm_params['afg']['ch1']['period'])
        delay0, delay1 = [self.Q_(val).to('s').magnitude for val in ivm_params['dg']['range']]
        delay_vec = np.linspace(delay0, delay1, ivm_params['dg']['nsteps'])
        vmod_vec = np.full_like(delay_vec, np.nan, dtype=np.double)

        #: Set AFG pulse parameters
        for ch in [1, 2]:
            p = ivm_params['afg']['ch{}'.format(ch)]
            getattr(self.afg1, 'voltage_high{}'.format(ch))('{}V'.format(self.Q_(p['high']).to('V').magnitude))
            getattr(self.afg1, 'voltage_low{}'.format(ch))('{}V'.format(self.Q_(p['low']).to('V').magnitude))
            getattr(self.afg1, 'pulse_period{}'.format(ch))('{}us'.format(self.Q_(p['period']).to('us').magnitude))
            getattr(self.afg1, 'pulse_width{}'.format(ch))('{}us'.format(self.Q_(p['width']).to('us').magnitude))
            getattr(self.afg1, 'pulse_trans_lead{}'.format(ch))('{}us'.format(self.Q_(p['lead']).to('us').magnitude))
            getattr(self.afg1, 'pulse_trans_trail{}'.format(ch))('{}us'.format(self.Q_(p['trail']).to('us').magnitude))
            getattr(self.afg1, 'pulse_delay{}'.format(ch))('{}us'.format(self.Q_(p['delay']).to('us').magnitude))

        #: Set delay generator parameters
        p = ivm_params['dg']
        self.dg.delay_B('A, {:e}'.format(delay0))
        self.dg.delay_C('T0, {:e}'.format(self.Q_(p['ch2']['delay']).to('s').magnitude))
        self.dg.delay_D('C, {:e}'.format(self.Q_(p['ch2']['width']).to('s').magnitude))

        self.dg.amp_out_AB(self.Q_(p['ch1']['voltage']).to('V').magnitude)
        self.dg.offset_out_AB(self.Q_(p['ch1']['offset']).to('V').magnitude)
        self.dg.amp_out_CD(self.Q_(p['ch2']['voltage']).to('V').magnitude)
        self.dg.offset_out_CD(self.Q_(p['ch2']['offset']).to('V').magnitude)

        self.MAG_lockin.time_constant(self.Q_(ivm_params['time_constant']).to('s').magnitude)

        #: Get instrument metadata and prefactors
        lockin_snap = self.MAG_lockin.snapshot(update=True)
        lockin_meta = {}
        for param in ['time_constant', 'sensitivity', 'phase', 'reserve', 'filter_slope']:
            lockin_meta.update({param: lockin_snap['parameters'][param]})
        meta_dict.update({'metadata':
                            {'lockin': lockin_meta,
                             'afg': self.afg1.snapshot(update=True),
                             'dg': self.dg.snapshot(update=True),
                             'ivm_params': ivm_params}
                        })
        #prefactor = 1 / (10 / lockin_snap['parameters']['sensitivity']['value'])
        #prefactor /= ivm_params['channels']['lockinX']['gain']

        with nidaqmx.Task('ai_task') as ai_task, nidaqmx.Task('ao_task') as ao_task:
            ao = '{}/ao{}'.format(daq_config['name'], daq_config['channels']['analog_outputs']['mod'])
            ao_task.ao_channels.add_ao_voltage_chan(ao, 'mod')
            for ch, idx in channels.items():
                channel = '{}/ai{}'.format(daq_config['name'], idx)
                ai_task.ai_channels.add_ai_voltage_chan(channel, ch)

            figM = plt.figure(figsize=(4,3))
            axM  = plt.gca()
            plt.xlim(min(delay_vec), max(delay_vec))
            plt.xlabel(r'Delay time [$\mu$s]')
            plt.ylabel('Modulation Voltage [V]')

            figT = plt.figure(figsize=(4,3))
            axT = plt.gca()
            plt.xlabel('Iteration number')
            plt.ylabel('Modulation Voltage [V]')

            log.info('Starting iv_tek_mod_daq.')
            try:
                #: Sweep delay time
                for j in range(len(delay_vec)):
                    self.dg.delay_A('T0, {:e}'.format(delay_vec[j]))
                    self.dg.delay_B('A, {:e}'.format(period.to('s').magnitude - delay_vec[j]))
                    elapsed_time = 0
                    nsamples = 0
                    vmod_time = np.array([])
                    t0 = time.time()
                    time.sleep(0.01)
                    #: Do digital PID control
                    while elapsed_time < tsettle + tavg:
                        ai_data = ai_task.read()
                        vcomp = ai_data[0]
                        err = vcomp - vcomp_set
                        vmod += P * err
                        vmod = np.mod(vmod, vmod_high)
                        #vmod = max(vmod, vmod_low) if vmod < 0 else min(vmod, vmod_high)
                        ao_task.write(vmod)
                        elapsed_time = time.time() - t0
                        nsamples += 1
                        vmod_time = np.append(vmod_time, vmod)
                    avg_start_pt = int(nsamples * tavg // (tsettle + tavg))
                    vmod_vec[j] = np.mean(vmod_time[avg_start_pt:])

                    utils.clear_artists(axM)
                    axM.plot(delay_vec, vmod_vec, 'bo-')
                    plt.tight_layout()
                    figM.canvas.draw()

                    utils.clear_artists(axT)
                    axT.plot(vmod_time, 'bo')
                    plt.tight_layout()
                    figT.canvas.draw()

                    time.sleep(0.05)
            except KeyboardInterrupt:
                log.warning('Measurement aborted by user.')

        data_dict.update({
            'delay_vec': {'array': delay_vec, 'unit': 's'},
            'vmod_vec': {'array': vmod_vec, 'unit': 'V'}
            })
        if ivm_params['save']:
            #: Get/create data location
            loc_provider = qc.FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
            loc = loc_provider(DiskIO('.'), record={'name': ivm_params['fname']})
            pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
            #: Save arrays to mat
            io.savemat('{}/{}'.format(loc, ivm_params['fname']), data_dict)
            #: Save metadata to json
            with open(loc + '/metadata.json', 'w') as f:
                try:
                    json.dump(meta_dict, f, sort_keys=True, indent=4, skipkeys=True)
                except TypeError:
                    pass
            #: Save figures to png
            figM.suptitle(loc)
            figM.tight_layout(rect=[0, 0.03, 1, 0.95])
            figM.savefig('{}/{}mod_d.png'.format(loc, ivm_params['fname']))
            figT.suptitle(loc)
            figT.tight_layout(rect=[0, 0.03, 1, 0.95])
            figT.savefig('{}/{}mod_t.png'.format(loc, ivm_params['fname']))
            log.info('Data saved to {}.'.format(loc))

        return data_dict, meta_dict

    def scan_surface(self, scan_params: Dict[str, Any]) -> None:
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
            log.warning('Scan interrupted by user. Going to [0, 0, 0] V.')
            #: Stop 'scan_plane_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #: If there's an active AO task, close it so that we can use goto
            try:
                self.scanner.control_ao_task('stop')
                self.scanner.control_ao_task('close')
            except:
                pass
            self.scanner.goto([0, 0, 0])
            #self.CAP_lockin.amplitude(0.004)
            #self.SUSC_lockin.amplitude(0.004)
            log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
        utils.scan_to_mat_file(data, real_units=True, interpolator=self.scanner.surface_interp)
        #return data, scan_plot
