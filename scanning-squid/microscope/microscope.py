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
import os
import sys
import time
import json
import pathlib
from typing import Dict, List, Sequence, Any, Union, Tuple
from collections import OrderedDict

#: Plotting and math modules
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as colors
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import Rbf
from scipy import io
from IPython.display import clear_output

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcodes.station import Station
from qcodes.instrument_drivers.stanford_research.SR830 import SR830
from qcodes.data.io import DiskIO

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
import squids
import instruments.atto as atto
import utils
from scanner import Scanner
from dclocfc import Dclocfc
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot, RPlot, RvTPlot, MSvTPlot
from instruments.lakeshore import Model_372, Model_331, Model_340
from instruments.keithley import Keithley_2400
from instruments.heater import EL320P

#: Pint for manipulating physical units
from pint import UnitRegistry
ureg = UnitRegistry()
#: Tell UnitRegistry instance what a Phi0 is, that Ohm = ohm, and what percent or pct is
with open('squid_units.txt', 'w') as f:
    f.write('Phi0 = 2.067833831e-15 * Wb\n')
    f.write('Ohm = ohm\n')
    f.write('Ohms = ohm\n')
    f.write('ohms = ohm\n')
    f.write('fraction = [] = frac\n')
    f.write('percent = 1e-2 frac = pct\n')
ureg.load_definitions('./squid_units.txt')


import logging
log = logging.getLogger(__name__)


class Microscope(Station):
    """Base class for scanning SQUID microscope.
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
        super().__init__(**kwargs)
        qc.Instrument.close_all()
        self.config = utils.load_json_ordered(config_file)
        if not os.path.exists('logs'):
            os.mkdir('logs')
        if log_name is None:
            log_file = utils.next_file_name('./logs/' + config_file.split('.')[0], 'log')
        else:
            log_file = utils.next_file_name('./logs/' + log_name, 'log')
        logging.basicConfig(
        	level=logging.INFO,
        	format='%(levelname)s:%(asctime)s:%(module)s:%(message)s',
            datefmt=self.config['info']['timestamp_format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ])
        log.info('Logging started.')
        log.info('Initializing microscope object using file {}.'.format(config_file))

        self.ureg = ureg
        # Callable for converting a string into a quantity with units
        self.Q_ = ureg.Quantity
        self.temp = temp

        self._add_atto()
        self._add_ls331()
        #self._add_ls372()
        self._add_ls340()
        #self._add_keithley()
        self._add_scanner()
        self._add_dclocfc()
        self._add_SQUID()
        self._add_lockins()
        self._add_ke2400()

    def _add_ke2400(self):
    	ke_config = self.config['instruments']['ke2400']
    	if hasattr(self,'ke2400'):
    		self.ke2400.close()
    	self.remove_component(ke_config['name'])
    	self.ke2400 = Keithley_2400(ke_config['name'], ke_config['address'])
    	self.add_component(self.ke2400)
    	log.info('Keithley 2400 successfully added to microscope')

    def _add_atto(self):
        """Add Attocube controller to microscope.
        """
        atto_config = self.config['instruments']['atto']
        ts_fmt = self.config['info']['timestamp_format']
        if hasattr(self, 'atto'):
        #     self.atto.clear_instances()
            self.atto.close()
        self.remove_component(atto_config['name'])
        self.atto = atto.ANC300(atto_config, self.temp, self.ureg, ts_fmt)
        self.add_component(self.atto)
        log.info('Attocube controller successfully added to microscope.')

    def _add_ls372(self):
        """Add Lakeshore 372 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls372']
        if hasattr(self, 'ls372'):
        #     self.atto.clear_instances()
            self.ls372.close()
        self.remove_component(ls_config['name'])
        self.ls372 = Model_372(ls_config['name'], ls_config['address'])
        self.add_component(self.ls372)
        log.info('Lakeshore 372 successfully added to microscope.')

    def _add_ls331(self):
        """Add Lakeshore 331 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls331']
        if hasattr(self, 'ls331'):
        #     self.atto.clear_instances()
            self.ls331.close()
        self.remove_component(ls_config['name'])
        self.ls331 = Model_331(ls_config['name'], ls_config['address'], ls_config['loop'],ls_config['Tmin_K'],ls_config['Tmax_K'])
        self.add_component(self.ls331)
        log.info('Lakeshore 331 successfully added to microscope.')

    def _add_ls340(self):
        """Add Lakeshore 340 temperature controller to microscope.
        """
        ls_config = self.config['instruments']['ls340']
        if hasattr(self, 'ls340'):
        #     self.atto.clear_instances()
            self.ls340.close()
        self.remove_component(ls_config['name'])
        self.ls340 = Model_340(ls_config['name'], ls_config['address'])
        self.add_component(self.ls340)
        log.info('Lakeshore 340 successfully added to microscope.')

    def _add_scanner(self):
        """Add scanner instrument to microscope.
        """
        scanner_config = self.config['instruments']['scanner']
        daq_config = self.config['instruments']['daq']
        if hasattr(self, 'scanner'):
            #self.scanner.clear_instances()
            self.scanner.close()
        self.remove_component(scanner_config['name'])
        self.scanner = Scanner(scanner_config, daq_config, self.temp, self.ureg)
        self.add_component(self.scanner)
        log.info('Scanner successfully added to microscope.')

    def _add_dclocfc(self):
        """Add dc local field through FC instrument to microscope.
        """
        dclocfc_config = self.config['instruments']['dclocfc']
        daq_config = self.config['instruments']['daq']
        if hasattr(self, 'dclocfc'):
            #self.dclocfc.clear_instances()
            self.dclocfc.close()
        self.remove_component(dclocfc_config['name'])
        self.dclocfc = Dclocfc(dclocfc_config, daq_config, self.temp, self.ureg)
        self.add_component(self.dclocfc)
        log.info('dclocfc successfully added to microscope.')
    
    def _add_SQUID(self):
        """Add SQUID instrument to microscope.
        """
        squid_config = self.config['SQUID']
        if hasattr(self, 'SQUID'):
            #self.SQUID.clear_instances()
            self.SQUID.close()
        self.remove_component(squid_config['name'])
        squid_type = squid_config['type'].lower().capitalize()
        self.SQUID = getattr(sys.modules['squids'], squid_type)(squid_config)
        self.add_component(self.SQUID)
        log.info('{}(SQUID) successfully added to microscope.'.format(squid_type))
        
    def _add_lockins(self):
        """Add lockins to microscope.
        """
        for lockin, lockin_info in self.config['instruments']['lockins'].items():
            name = '{}_lockin'.format(lockin)
            address = lockin_info['address']
            if hasattr(self, name):
                #getattr(self, name, 'clear_instances')()
                getattr(self, name, 'close')()
            self.remove_component(name)
            instr = SR830(name, address, metadata={lockin: lockin_info})
            setattr(self, name, instr)
            self.add_component(getattr(self, '{}_lockin'.format(lockin)))
            log.info('{} successfully added to microscope.'.format(name))
            
    def set_lockins(self, measurement: Dict[str, Any]) -> None:
        """Initialize lockins for given measurement.

        Args:
            measurement: Dict of measurement parameters as defined
                in measurement configuration file.
        """
        channels = measurement['channels']
        for ch in channels:
            if 'lockin' in channels[ch]:
                lockin = '{}_lockin'.format(channels[ch]['lockin']['name'])
                for param in channels[ch]['lockin']:
                    if param != 'name':
                        parameters = getattr(self, lockin).parameters
                        unit = parameters[param].unit
                        value = self.Q_(channels[ch]['lockin'][param]).to(unit).magnitude
                        log.info('Setting {} on {} to {} {}.'.format(param, lockin, value, unit))
                        parameters[param].set(value)
        time.sleep(1)

    def td_cap(self, tdc_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a capacitive touchdown.

        Args:
            tdc_params: Dict of capacitive touchdown parameters as defined
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)

        Returns:
            Tuple[qcodes.DataSet, plots.TDCPlot]: data, tdc_plot
                DataSet and plot generated by the touchdown Loop.
        """
        old_pos = self.scanner.position()
        constants = tdc_params['constants']
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = tdc_params['channels']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels
        self.set_lockins(tdc_params)
        self.snapshot(update=update_snap)
        #: z position voltage step
        dV = self.Q_(tdc_params['dV']).to('V').magnitude
        #: Start and end z position voltages
        startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in tdc_params['range']])

        # for susceptometers:
        #delay = constants['wait_factor'] * max(self.CAP_lockin.time_constant(), self.SUSC_lockin.time_constant())

        # for samplers:
        delay = constants['wait_factor'] * max(self.CAP_lockin.time_constant(), self.SUSC_lockin.time_constant())
        
        prefactors = self.get_prefactors(tdc_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = tdc_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('td_cap_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            #self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        loop_counter = utils.Counter()
        tdc_plot = TDCPlot(tdc_params, self.ureg) 
        loop = qc.Loop(self.scanner.position_z.sweep(startV, endV, dV)
            ).each(
                qc.Task(time.sleep, delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.check_for_td, tdc_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(self.scanner.get_td_height, tdc_plot),
                qc.BreakIf(lambda scanner=self.scanner: scanner.break_loop or scanner.td_has_occurred),
                qc.Task(loop_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004),
                qc.Task(self.scanner.retract),
                qc.Task(tdc_plot.fig.show),
                qc.Task(tdc_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(tdc_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=tdc_params['fname'], write_period=None)
        try:
            log.info('Starting capacitive touchdown.')
            loop.run()
            if self.scanner.td_height is not None:
                data.metadata['loop']['metadata'].update({'td_height': self.scanner.td_height})
                if abs(old_pos[0]) < 0.01 and abs(old_pos[1]) < 0.01:
                    self.scanner.metadata['plane'].update({'z': self.scanner.td_height})
        except KeyboardInterrupt:
            log.warning('Touchdown interrupted by user. Retracting scanner. DataSet saved to {}.'.format(data.location))
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
        finally:
            try:
                #: Stop 'td_cap_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                self.scanner.retract()
                #self.CAP_lockin.amplitude(0.004)
                tdc_plot.fig.show()
                tdc_plot.save()
            except:
                pass
        utils.td_to_mat_file(data, real_units=True)
        return data, tdc_plot

    def measure_R(self, R_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a four prob measurement

        Args:
            R_params: Dict of 4 prob parameters as defined
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)

        Returns:
            Tuple[qcodes.DataSet, plots.Four_prob]: data, Four_prob
                DataSet and plot.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = R_params['channels']
        constants = R_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude

        self.set_lockins(R_params)
        self.snapshot(update=update_snap)
         #: out put delta V
        dV = self.Q_(R_params['dV']).to('V').magnitude
        #: bias voltage output range
        startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in R_params['range']])
        if startV == .04:
            startV = dV

        delay = constants['wait_factor'] * self.Four_prob_lockin.time_constant()
        
        prefactors = self.get_prefactors(R_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = R_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('Four_prob_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            #self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        loop_counter = utils.Counter()
        R_plot = RPlot(R_params, self.ureg) 
        self.SUSC_lockin.amplitude(startV)
        print(startV)
        print(dV)
        loop = qc.Loop(self.Four_prob_lockin.amplitude.sweep(startV, endV, dV)
            ).each(
                qc.Task(time.sleep, delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.get_R, R_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(loop_counter.advance)
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004),
                qc.Task(R_plot.fig.show),
                qc.Task(R_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(R_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=R_params['fname'], write_period=None)
        try:
            log.info('Starting voltage sweep:')
            loop.run()
        except KeyboardInterrupt:
            log.warning('Four prob measurement interrupted by user. DataSet saved to {}.'.format(data.location))
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
            self.Four_prob_lockin.amplitude(0.04)
            ai_task.stop()
            ai_task.close()
        finally:
            try:
                #: Stop 'td_cap_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                self.Four_prob_lockin.amplitude(0.04)
                R_plot.fig.show()
                R_plot.save()
            except:
                pass
        utils.R_to_mat_file(data, real_units=True)

        return data, R_plot

    def measure_R_v_T(self, R_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs a four prob measurement over a range of temperature

        Args:
            R_params: Dict of 4 prob parameters as defined
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)

        Returns:
            Tuple[qcodes.DataSet, plots.Four_prob]: data, Four_prob
                DataSet and plot.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = R_params['channels']
        constants = R_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude

        self.set_lockins(R_params)
        self.snapshot(update=update_snap)
         #: out put delta V
        dT = self.Q_(constants['dT']).to('K').magnitude
        #: bias voltage output range
        ramp_rate = self.ls331.ramp_rate()
        ramp_rate = ramp_rate.split(',')[1]
        ramp_rate = float(ramp_rate)
        startT, endT = sorted([self.Q_(lim).to('K').magnitude for lim in constants['T_range']])


        delay = constants['wait_factor'] * self.Four_prob_lockin.time_constant()
        T_delay = constants['T_wait_factor'] * dT/ramp_rate * 60
        print("data acquization time interval is(s)")
        print(T_delay)

        prefactors = self.get_prefactors(R_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = R_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('Four_prob_v_T_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        loop_counter = utils.Counter()
        RvT_plot = RvTPlot(R_params, self.ureg) 
        loop = qc.Loop(self.ls331.set_temperature.sweep(startT, endT, dT)
            ).each(
                qc.Task(time.sleep, T_delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.get_RvT, RvT_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(loop_counter.advance),
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004),
                qc.Task(RvT_plot.fig.show),
                qc.Task(RvT_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(R_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=R_params['fvTname'], write_period=None)
        try:
            log.info('Starting voltage sweep:')
            loop.run()
        except KeyboardInterrupt:
            log.warning('Four prob measurement interrupted by user. DataSet saved to {}.'.format(data.location))
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
            self.Four_prob_lockin.amplitude(0.04)
            ai_task.stop()
            ai_task.close()
        finally:
            try:
                #: Stop 'td_cap_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                self.Four_prob_lockin.amplitude(0.04)
                RvT_plot.fig.show()
                RvT_plot.save()
            except:
                pass
        utils.RvT_to_mat_file(data, real_units=True)

        return data, RvT_plot


    def temp_series(self, temps_params: Dict[str, Any], update_snap: bool=True) -> Tuple[Any]:
        """Performs MAG and SUSCX measurements over a range of temperature

        Args:
            temps_params: Dict of temperature range and scan parameters as defined
                in measurement configuration file.
            update_snap: Whether to update the microscope snapshot. Default True.
                (You may want this to be False when getting a plane or approaching.)

        Returns:
            Tuple[qcodes.DataSet, plots.temp_series]: data, MSvTPlot
                DataSet and plot.
        """
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = temps_params['channels']
        Tramp_rate = self.Q_(temps_params['ramp_rate']).to('K/min').magnitude
        constants = temps_params['constants']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        # daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude
        daq_rate = self.Q_(constants['daq_rate']).to('Hz').magnitude
        sampleduration = self.Q_(constants['sampleduration']).to('sec').magnitude
        nsamples = int(daq_rate * sampleduration)
        

        self.set_lockins(temps_params)
        self.snapshot(update=update_snap)
        #: delta T
        dT = self.Q_(temps_params['dT']).to('K').magnitude
        #: temperature range
        self.ls331.ramp_rate(Tramp_rate)
        ramp_rate = self.ls331.ramp_rate()
        ramp_rate = ramp_rate.split(',')[1]
        ramp_rate = float(ramp_rate)
        startT, endT = [self.Q_(lim).to('K').magnitude for lim in temps_params['range']]


        delay = constants['wait_factor'] * self.SUSC_lockin.time_constant()
        T_delay = constants['T_wait_factor'] * abs(dT)/ramp_rate * 60 + delay
        # print("data acquization time interval is(s)")
        # print(T_delay)

        prefactors = self.get_prefactors(temps_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = temps_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task =  nidaqmx.Task('MAG_SUSCX_v_T_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
            self.daq_ai.close()
        # self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task)
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,samples_to_read=nsamples,target_points=1, timeout=sampleduration+1)
        loop_counter = utils.Counter()
        MSvT_plot = MSvTPlot(temps_params, self.ureg) 
        loop = qc.Loop(self.ls331.set_temperature.sweep(startT, endT, dT)
            ).each(
                qc.Task(time.sleep, T_delay),
                self.daq_ai.voltage,
                qc.Task(self.scanner.get_MSvT, MSvT_plot, qc.loops.active_data_set, loop_counter),
                qc.Task(loop_counter.advance),
            ).then(
                qc.Task(ai_task.stop),
                qc.Task(ai_task.close),
                #qc.Task(self.CAP_lockin.amplitude, 0.004),
                #qc.Task(self.SUSC_lockin.amplitude, 0.004),
                qc.Task(MSvT_plot.fig.show),
                qc.Task(MSvT_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(temps_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=temps_params['fname'], write_period=None)
        try:
            log.info('Starting temperature sweep:')
            loop.run()
        except KeyboardInterrupt:
            log.warning('Temperature series interrupted by user. DataSet saved to {}.'.format(data.location))
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
            ai_task.stop()
            ai_task.close()
        finally:
            try:
                #: Stop 'MAG_SUSCX_v_T_ai_task' so that we can read our current position
                ai_task.stop()
                ai_task.close()
                MSvT_plot.fig.show()
                MSvT_plot.save()
            except:
                pass
        utils.MSvT_to_mat_file(data, real_units=True)

        return data, MSvT_plot



    def approach(self, tdc_params: Dict[str, Any], attosteps: int=100) -> None:
        """Approach the sample by iteratively stepping z Attocube and performing td_cap().

        Args:
            tdc_params: Dict of capacitive touchdown parameters as defined
                in measurement configuration file.
            attosteps: Number of z atto steps to perform per iteration. Default 100.
        """
        self.snapshot(update=True)
        log.info('Attempting to approach sample.')
        #: Perform an initial touchdown to make sure we're not close to the sample.
        data, tdc_plot = self.td_cap(tdc_params, update_snap=False)
        plt.close(tdc_plot.fig)
        clear_output(wait=True)
        while not self.scanner.break_loop:
            self.atto.step('z', attosteps)
            data, tdc_plot = self.td_cap(tdc_params, update_snap=False)
            plt.close(tdc_plot.fig)
            clear_output(wait=True)
        if self.scanner.td_has_occurred:
            log.info('Touchdown detected. Performing td_cap() to confirm.')    
            data, tdc_plot = self.td_cap(tdc_params, update_snap=False)
            if not self.scanner.td_has_occurred:
                log.warning('Could not confirm touchdown.')
            else:
                log.info('Touchdown confirmed.')

    def get_surface(self, x_vec: np.ndarray, y_vec: np.ndarray,
                    tdc_params: Dict[str, Any]) -> None:
        """Performs touchdowns on a grid and fits a plane to the resulting surface.

        Args:
            x_vec: 1D array of x positions (must be same length as y_vec).
            y_vec: 1D array of y positions (must be same length as x_vec).
            tdc_params: Dict of capacitive touchdown parameters as defined
                in measurement configuration file.

        Returns:
            Tuple[Union[np.ndarray, None]]: x_grid, y_grid, td_grid, plane
                x, y, td grids and plane coefficients such that td_grid is
                fit by x_grid * plane[0] + ygrid * plane[1] + plane[2].

        """
        old_pos = self.scanner.position()
        #: True if touchdown doesn't occur for any point in the grid
        out_of_range = False
        #: True if the loop is exited before finishing
        premature_exit = False
        self.scanner.break_loop = False
        self.scanner.td_has_occurred = False
        self.snapshot(update=True)
        x_grid, y_grid = np.meshgrid(x_vec, y_vec, indexing='ij')
        td_grid = np.full((len(x_vec), len(y_vec)), np.nan, dtype=np.double)
        log.info('Aqcuiring a plane.')
        v_retract = self.scanner.voltage_retract[self.temp].to('V').magnitude
        fig = plt.figure(figsize=(8,3))
        ax0 = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122, projection='3d')
        for ax in [ax0, ax1]:
            ax.set_xlabel('x position [V]')
            ax.set_ylabel('y position [V]')
            ax.set_zlabel('z position [V]')
        ax0.set_title('Sample Plane')
        ax1.set_title('Sample Surface') 
        fig.canvas.draw()  
        fig.show()
        for i in range(len(x_vec)):
            for j in range(len(y_vec)):
                #: If any of the safety limits in td_cap() are exceeded,
                #: or the loop is interrupted by the user.
                if self.scanner.break_loop and not self.scanner.td_has_occurred:
                    log.warning('Aborting get_surface().')
                    premature_exit = True
                    break #: goes to outer break statement
                else:
                    self.scanner.goto([x_grid[i,j], y_grid[i,j], v_retract])
                    data, tdc_plot = self.td_cap(tdc_params, update_snap=False)
                    td_grid[i,j] = self.scanner.td_height
                    clear_output(wait=True)
                    if self.scanner.td_height is None:
                        out_of_range = True
                        premature_exit = True
                        log.warning('Touchdown out of range. Stopping get_surface().')
                        self.scanner.goto(old_pos)
                        break #: goes to outer break statement
                    plt.close(fig)
                    fig = plt.figure(figsize=(8,3))
                    ax0 = fig.add_subplot(121, projection='3d')
                    ax1 = fig.add_subplot(122, projection='3d')
                    for ax in [ax0, ax1]:
                        ax.scatter(x_grid[np.isfinite(td_grid)], y_grid[np.isfinite(td_grid)],
                            td_grid[np.isfinite(td_grid)], cmap='viridis')
                        ax.set_xlabel('x position [V]')
                        ax.set_ylabel('y position [V]')
                        ax.set_zlabel('z position [V]')
                    ax0.set_title('Sample Plane')
                    ax1.set_title('Sample Surface') 
                    fig.canvas.draw()  
                    fig.show()
                    plt.close(tdc_plot.fig)
                    continue #: skips outer break statement
                break #: occurs only if out_of_range or loop is broken
        self.scanner.goto(old_pos)
        if not out_of_range and not premature_exit:
            self.scanner.metadata.update({'td_grid': {'x': x_grid, 'y': y_grid, 'z': td_grid}})
            # Create spline function to interpolate over surface:
            self.scanner.surface_interp = Rbf(x_grid, y_grid, td_grid, function='cubic')
            #: Fit a plane to the td_grid
            x = np.reshape(x_grid, (-1, 1))
            y = np.reshape(y_grid, (-1, 1))
            td = np.reshape(td_grid, (-1, 1))
            z = np.column_stack((x, y, np.ones_like(x)))
            plane, res, _, _ = lstsq(z, td)
            log.info('New plane : {}.'.format([plane[i][0] for i in range(3)]))
            ax0.plot_surface(x_grid, y_grid, plane[0] * x_grid + plane[1] * y_grid + plane[2],
                cmap='viridis', alpha=0.5)
            ax1.plot_surface(x_grid, y_grid, self.scanner.surface_interp(x_grid, y_grid),  cmap='viridis', alpha=0.5)
            for i, axis in enumerate(['x', 'y', 'z']):
                self.scanner.metadata['plane'].update({axis: plane[i][0]})
            self.atto.surface_is_current = True
            loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
            loc = loc_provider(DiskIO('.'), record={'name': 'surface'})
            pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
            fig.suptitle(loc)
            fig.canvas.draw()
            fig.show()
            plt.savefig(loc + '/surface.png')
            mdict = {
                'plane': {ax: self.scanner.metadata['plane'][ax] for ax in ['x', 'y', 'z']},
                'td_grid': {ax: self.scanner.metadata['td_grid'][ax] for ax in ['x', 'y', 'z']}
            }
            io.savemat(loc + '/surface.mat', mdict)
            #return x_grid, y_grid, td_grid, plane
        #: If the loop didn't finish, return (None, None, None, None)
        #return (None,) * 4
 
    def remove_component(self, name: str) -> None:
        """Remove a component (instrument) from the microscope.

        Args:
            name: Name of component to remove.
        """
        if name in self.components:
            _ = self.components.pop(name)
            log.info('Removed {} from microscope.'.format(name))
        else:
            log.debug('Microscope has no component with the name {}'.format(name))  



    def pulse_field(self, plsf_params: Dict[str, Any]):
        """Records a time trace of data from DAQ analog input channels, converts data to desired units.

        Args:
            samplerate: DAQ sampling rate (for each channel) in Hz.
            sampleduration1: Sampling time in seconds while applying fieldcurrent1
            sampleduration2: Sampling time in seconds while applying fieldcurrent2
            
        Returns:
            Dict: mdict
        """
        sampleduration1 = self.Q_(plsf_params['sampleduration1']).to('sec').magnitude
        sampleduration2 = self.Q_(plsf_params['sampleduration2']).to('sec').magnitude
        sampleduration = sampleduration1 +  sampleduration2
        fieldcurrent1 = self.Q_(plsf_params['fieldcurrent1']).to('A').magnitude
        fieldcurrent2 = self.Q_(plsf_params['fieldcurrent2']).to('A').magnitude
        samplerate = self.Q_(plsf_params['samplerate']).to('Hz').magnitude
        loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
        loc = loc_provider(DiskIO('.'), record={'name': 'pulse_field'})
        pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = plsf_params['channels']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        prefactors = self.get_prefactors(plsf_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        units = {}
        for ch, prefac in prefactors.items():
            unit = plsf_params['channels'][ch]['unit']
            units[ch] = unit
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        nsamples1 = int(samplerate * sampleduration1)
        nsamples2 = int(samplerate * sampleduration2)
        time = np.linspace(0, sampleduration1+sampleduration2, nsamples1+nsamples2)
        field1 = np.linspace(0, sampleduration1, nsamples1)*0+fieldcurrent1
        field2 = np.linspace(0, sampleduration2, nsamples2)*0+fieldcurrent2
        field = np.append(field1,field2)
        mdict = {
            'time': {'array': time, 'unit': 's'},
            'field': {'array': field, 'unit': 'A'},
            'metadata': {
                'samplerate': samplerate,
                'sampleduration': sampleduration,
                'location': loc
            }
        }

        log.info('Applying Magnetic field {} A for {} sec, then quanching to {} A'.format(fieldcurrent1,sampleduration1,fieldcurrent2))
        log.info('Start pluse field time trace of temperature')
        with nidaqmx.Task('pulse_field_ai_task1') as ai_task1:
            self.ke2400.rangei(fieldcurrent1)
            self.ke2400.curr(fieldcurrent1)
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', daq_name, samplerate, channels, ai_task1,
                                     samples_to_read=nsamples1, timeout=sampleduration1+10)
            data_v1 = daq_ai.voltage()
            daq_ai.close()
        with nidaqmx.Task('pulse_field_ai_task2') as ai_task2:
            self.ke2400.rangei(fieldcurrent2)
            self.ke2400.curr(fieldcurrent2)
            daq_ai = DAQAnalogInputs('daq_ai', daq_name, samplerate, channels, ai_task2,
                                     samples_to_read=nsamples2, timeout=sampleduration2+10)
            data_v2 = daq_ai.voltage()
            daq_ai.close()
        for i, ch in enumerate(channels):
            mdict.update({ch: {'array': np.append(data_v1[i],data_v2[i]) * prefactors[ch].magnitude, 'unit': units[ch], 'prefactor': prefactor_strs[ch]}})
        
        fig, ax = plt.subplots(2, figsize=(8,4), tight_layout=True, sharex=True)
        ax[0].plot(time, np.append(data_v1[0],data_v2[0]) *prefactors['TEMP'].magnitude, lw=1)
        ax[0].set_ylabel(plsf_params['channels']['TEMP']['label'] + plsf_params['channels']['TEMP']['unit_latex'])
        ax[0].grid()
        ax[1].plot(time, field, lw=1)
        ax[1].set_ylabel('Field [A]')
        fig.suptitle(loc, x=0.5, y=1)
        ax[1].set_xlabel('Time [sec]')
        ax[1].grid()
        
        plt.savefig(loc + '/pulse_field.png')
        io.savemat(loc + '/pulse_field.mat', mdict)
        return mdict


    def pulse_heater(self, plsh_params: Dict[str, Any]):
        """Records a time trace of data from DAQ analog input channels, converts data to desired units.

        Args:
            samplerate: DAQ sampling rate (for each channel) in Hz.
            sampleduration1: Sampling time in seconds while applying manual heater power(mhp1)
            sampleduration2: Sampling time in seconds while applying manual heater power(mhp2)
            
        Returns:
            Dict: mdict
        """
        sampleduration1 = self.Q_(plsh_params['sampleduration1']).to('sec').magnitude
        sampleduration2 = self.Q_(plsh_params['sampleduration2']).to('sec').magnitude
        sampleduration = sampleduration1 +  sampleduration2
        mhp1 = self.Q_(plsh_params['mhp1']).to('pct').magnitude
        mhp2 = self.Q_(plsh_params['mhp2']).to('pct').magnitude
        samplerate = self.Q_(plsh_params['samplerate']).to('Hz').magnitude
        loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
        loc = loc_provider(DiskIO('.'), record={'name': 'pulse_field'})
        pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
        daq_config = self.config['instruments']['daq']
        daq_name = daq_config['name']
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = plsh_params['channels']
        channels = {} 
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        prefactors = self.get_prefactors(plsh_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        units = {}
        for ch, prefac in prefactors.items():
            unit = plsh_params['channels'][ch]['unit']
            units[ch] = unit
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        nsamples1 = int(samplerate * sampleduration1)
        nsamples2 = int(samplerate * sampleduration2)
        time = np.linspace(0, sampleduration1+sampleduration2, nsamples1+nsamples2)
        heater1 = np.linspace(0, sampleduration1, nsamples1)*0+mhp1
        heater2 = np.linspace(0, sampleduration2, nsamples2)*0+mhp2
        heater = np.append(heater1,heater2)
        mdict = {
            'time': {'array': time, 'unit': 's'},
            'mhp': {'array': heater, 'unit': '%'},
            'metadata': {
                'samplerate': samplerate,
                'sampleduration': sampleduration,
                'location': loc
            }
        }

        log.info('Applying manual heater power {} %  for {} sec, then quanching to {} %'.format(mhp1,sampleduration1,mhp2))
        log.info('Start pluse field time trace of temperature')
        with nidaqmx.Task('pulse_heater_ai_task1') as ai_task1:
            self.ls331.mhp(mhp1)
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', daq_name, samplerate, channels, ai_task1,
                                     samples_to_read=nsamples1, timeout=sampleduration1+10)
            data_v1 = daq_ai.voltage()
            daq_ai.close()
        with nidaqmx.Task('pulse_heater_ai_task2') as ai_task2:
            self.ls331.mhp(mhp2)
            daq_ai = DAQAnalogInputs('daq_ai', daq_name, samplerate, channels, ai_task2,
                                     samples_to_read=nsamples2, timeout=sampleduration2+10)
            data_v2 = daq_ai.voltage()
            daq_ai.close()
        for i, ch in enumerate(channels):
            mdict.update({ch: {'array': np.append(data_v1[i],data_v2[i]) * prefactors[ch].magnitude, 'unit': units[ch], 'prefactor': prefactor_strs[ch]}})
        
        fig, ax = plt.subplots(2, figsize=(8,4), tight_layout=True, sharex=True)
        ax[0].plot(time, np.append(data_v1[0],data_v2[0]) *prefactors['TEMP'].magnitude, lw=1)
        ax[0].set_ylabel(plsh_params['channels']['TEMP']['label'] + plsh_params['channels']['TEMP']['unit_latex'])
        ax[0].grid()
        ax[1].plot(time, heater, lw=1)
        ax[1].set_ylabel('Heater [%]')
        fig.suptitle(loc, x=0.5, y=1)
        ax[1].set_xlabel('Time [sec]')
        ax[1].grid()
        
        plt.savefig(loc + '/pulse_heater.png')
        io.savemat(loc + '/pulse_heater.mat', mdict)
        return mdict
            