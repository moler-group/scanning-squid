#: Various Python utilities
import os
import sys
import time
import json
from typing import Dict, List, Sequence, Any, Union, Tuple
from collections import OrderedDict

#: Plotting and math modules
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as colors
import numpy as np
from scipy.linalg import lstsq
from IPython.display import clear_output

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcodes.station import Station
#from qcodes.instrument_drivers.tektronix.Keithley_2400 import Keithley_2400
from instruments.keithley_2400 import Keithley_2400
from instruments.hf2li import HF2LI
from instruments.lakeshore import Model_335

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
import squids
import instruments.atto as atto
import utils
from scanner import Scanner
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot

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
        #with open(config_file) as f:
        #    self.config = json.load(f, object_pairs_hook=OrderedDict)
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

#         self._add_atto()
        self._add_temp_controller()
        self._add_keithley2440()
        self._add_keithley2400()
        self._add_keithley2400_1()
        self._add_scanner()
        self._add_SQUID()
        self._add_lockins()

    def _add_atto(self):
        """Add Attocube controller to microscope.
        """
        atto_config = self.config['instruments']['atto']
        ts_fmt = self.config['info']['timestamp_format']
        if hasattr(self, 'atto'):
            self.atto.clear_instances()
        self.remove_component(atto_config['name'])
        self.atto = atto.ANC300(atto_config, self.temp, self.ureg, ts_fmt)
        self.add_component(self.atto)
        log.info('Attocube controller successfully added to microscope.')

    def _add_temp_controller(self):
        """Add temperature controller to microscope.
        """
        tc_config = self.config['instruments']['temp_controller']
        if hasattr(self, 'temp_controller'):
            self.temp_controller.clear_instances()
        self.remove_component(tc_config['name'])
        self.temp_controller = Model_335(tc_config['name'], tc_config['address'])
        self.add_component(self.temp_controller)
        self.temp_controller.heater_range(0)
        self.temp_controller.ramp_rate(0)
        log.info('Temperature controller successfully added to microscope.')

    def _add_keithley2440(self):
        """Add Keithley to microscope.
        """
        k_config = self.config['instruments']['keithley2440']
        if hasattr(self, 'keithley2440'):
            self.keithley2440.clear_instances()
        self.remove_component(k_config['name'])
        self.keithley2440 = Keithley_2400(k_config['name'], k_config['address'])
        self.add_component(self.keithley2440)
        self.keithley2440.mode('CURR')
        #self.keithley.sense('VOLT')
        self.keithley2440.rangei(100e-3)
        self.keithley2440.curr(0)
        self.keithley2440.compliancev(10)
        self.keithley2440.output(1)
        log.info('Keithley 2440 controller successfully added to microscope.')

    def _add_keithley2400(self):
        """Add Keithley to microscope.
        """
        k_config = self.config['instruments']['keithley2400']
        if hasattr(self, 'keithley2400'):
            self.keithley_2400.clear_instances()
        self.remove_component(k_config['name'])
        self.keithley2400 = Keithley_2400(k_config['name'], k_config['address'])
        self.add_component(self.keithley2400)
        self.keithley2400.mode('CURR')
        #self.keithley.sense('VOLT')
        self.keithley2400.rangei(100e-3)
        self.keithley2400.curr(0)
        self.keithley2400.compliancev(10)
        self.keithley2400.output(1)
        log.info('Keithley 2400 controller successfully added to microscope.')
    
    def _add_keithley2400_1(self):
        """Add Keithley to microscope.
        """
        k_config = self.config['instruments']['keithley2400_1']
        if hasattr(self, 'keithley2400_1'):
            self.keithley2400_1.clear_instances()
        self.remove_component(k_config['name'])
        self.keithley2400_1 = Keithley_2400(k_config['name'], k_config['address'])
        self.add_component(self.keithley2400_1)
        self.keithley2400_1.mode('CURR')
        #self.keithley.sense('VOLT')
        self.keithley2400_1.rangei(100e-3)
        self.keithley2400_1.curr(0)
        self.keithley2400_1.compliancev(10)
        self.keithley2400_1.output(1)
        log.info('Keithley 2400 controller 2 successfully added to microscope.')

    def _add_scanner(self):
        """Add scanner instrument to microscope.
        """
        scanner_config = self.config['instruments']['scanner']
        daq_config = self.config['instruments']['daq']
        if hasattr(self, 'scanner'):
            self.scanner.clear_instances()
        self.remove_component(scanner_config['name'])
        self.scanner = Scanner(scanner_config, daq_config, self.temp, self.ureg)
        self.add_component(self.scanner)
        log.info('Scanner successfully added to microscope.')
    
    def _add_SQUID(self):
        """Add SQUID instrument to microscope.
        """
        squid_config = self.config['SQUID']
        if hasattr(self, 'SQUID'):
            self.SQUID.clear_instances()
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
            if hasattr(self, name):
                getattr(self, name, 'clear_instances')()
            self.remove_component(name)
            dev = lockin_info['device']
            demod = lockin_info['demod']
            sigout = lockin_info['sigout']
            auxouts = lockin_info['auxouts']
            instr = HF2LI(name, dev, demod, sigout, auxouts, metadata={lockin: lockin_info})
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
        #self.set_lockins(tdc_params)
        self.snapshot(update=update_snap)
        #: z position voltage step
        dV = self.Q_(tdc_params['dV']).to('V').magnitude
        #: Start and end z position voltages
        startV, endV = sorted([self.Q_(lim).to('V').magnitude for lim in tdc_params['range']])
        delay = constants['wait_factor'] * self.CAP_lockin.time_constant()
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
            self.daq_ai.clear_instances()
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
                qc.Task(self.scanner.retract),
                qc.Task(tdc_plot.fig.show),
                qc.Task(tdc_plot.save)
            )
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(tdc_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=tdc_params['fname'])
        try:
            log.info('Starting capacitive touchdown.')
            loop.run()
            if abs(old_pos[0]) < 0.002 and abs(old_pos[1]) < 0.002 and self.scanner.td_height is not None:
                self.scanner.metadata['plane'].update({'z': self.scanner.td_height})
        except KeyboardInterrupt:
            log.warning('Touchdown interrupted by user. Retracting scanner.')
            #: Set break_loop = True so that get_plane() and approach() will be aborted
            self.scanner.break_loop = True
            #: Stop 'td_cap_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            self.scanner.retract()
            #self.CAP_lockin.amplitude(0.004)
            tdc_plot.fig.show()
            tdc_plot.save()
            log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        utils.td_to_mat_file(data, real_units=True)
        return data, tdc_plot

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

    def get_plane(self, x_vec: np.ndarray, y_vec: np.ndarray,
        tdc_params: Dict[str, Any]) -> Tuple[Union[np.ndarray, None]]:
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
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x position [V]')
        ax.set_ylabel('y position [V]')
        ax.set_zlabel('z position [V]')
        ax.set_title('Sample Plane')  
        fig.canvas.draw()  
        fig.show()
        for i in range(len(x_vec)):
            for j in range(len(y_vec)):
                #: If any of the safety limits in td_cap() are exceeded,
                #: or the loop is interrupted by the user.
                if self.scanner.break_loop and not self.scanner.td_has_occurred:
                    log.warning('Aborting get_plane().')
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
                        log.warning('Touchdown out of range. Stopping get_plane().')
                        self.scanner.goto(old_pos)
                        break #: goes to outer break statement
                    plt.close(fig)
                    fig = plt.figure(figsize=(4,3))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(x_grid[np.isfinite(td_grid)], y_grid[np.isfinite(td_grid)],
                        td_grid[np.isfinite(td_grid)], cmap='viridis')
                    ax.set_xlabel('x position [V]')
                    ax.set_ylabel('y position [V]')
                    ax.set_zlabel('z position [V]')
                    ax.set_title('Sample Plane')  
                    fig.canvas.draw()  
                    fig.show()
                    plt.close(tdc_plot.fig)
                    continue #: skips outer break statement
                break #: occurs only if out_of_range or loop is broken
        self.scanner.goto(old_pos)
        if not out_of_range and not premature_exit:
            #: Fit a plane to the td_grid
            x = np.reshape(x_grid, (-1, 1))
            y = np.reshape(y_grid, (-1, 1))
            td = np.reshape(td_grid, (-1, 1))
            z = np.column_stack((x, y, np.ones_like(x)))
            plane, _, _, _ = lstsq(z, td)
            log.info('New plane : {}.'.format([plane[i][0] for i in range(3)]))
            ax.plot_surface(x_grid, y_grid, plane[0] * x_grid + plane[1] * y_grid + plane[2],
                cmap='viridis', alpha=0.5)
            ax.set_title(data.location)
            fig.canvas.draw()
            fig.show()
            plt.savefig(data.location + '/plane.png')
            for i, axis in enumerate(['x', 'y', 'z']):
                self.scanner.metadata['plane'].update({axis: plane[i][0]})
            #self.atto.plane_is_current = True
            return x_grid, y_grid, td_grid, plane
        #: If the loop didn't finish, return (None, None, None, None)
        return (None,) * 4
 
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
            