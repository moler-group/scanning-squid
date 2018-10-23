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

#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
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
            prefactor /= measurement['channels'][ch]['gain']
            prefactors.update({ch: prefactor.to('{}/V'.format(measurement['channels'][ch]['unit']))})
        return prefactors

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
