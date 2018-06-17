import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
from utils import fit_line
from typing import Dict, List, Optional, Sequence, Any, Union
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode
import logging
log = logging.getLogger(__name__)

class Scanner(Instrument):
    """Controls DAQ AOs to drive the scanner.
    """   
    def __init__(self, scanner_config: Dict[str, Any], daq_config: Dict[str, Any],
                 temp: str, ureg: Any, **kwargs) -> None:
        """
        Args:
            scanner_config: Scanner configuration dictionary as defined
                in microscope configuration JSON file.
            daq_config: DAQ configuration dictionary as defined
                in microscope configuration JSON file.
            temp: 'LT' or 'RT' - sets the scanner voltage limit for each axis
                based on temperature mode.
            ureg: pint UnitRegistry, manages units.
        """
        super().__init__(scanner_config['name'], **kwargs)
        if temp.upper() not in ['LT', 'RT']:
            raise ValueError('Temperature mode must be "LT" or "RT".')
        self.temp = temp.upper()
        self.ureg = ureg
        self.Q_ = ureg.Quantity
        self.metadata.update(scanner_config)
        self.metadata['position'].update({'plane_is_current': False})
        self.metadata.update({'daq': daq_config})
        self._parse_unitful_quantities()
        self._initialize_parameters()
        self.goto([0, 0, 0])
        
    def _parse_unitful_quantities(self):
        """Parse strings from configuration dicts into Quantities with units.
        """
        self.daq_rate = self.Q_(self.metadata['daq']['rate']).to('Hz').magnitude
        self.voltage_retract = {'RT': self.Q_(self.metadata['voltage_retract']['RT']),
                                'LT': self.Q_(self.metadata['voltage_retract']['LT'])}
        self.speed = self.Q_(self.metadata['speed']['value'])
        self.constants = {'comment': self.metadata['constants']['comment']}
        self.voltage_limits = {'RT': {},
                               'LT': {},
                               'unit': self.metadata['voltage_limits']['unit'],
                               'comment': self.metadata['voltage_limits']['comment']}
        unit = self.voltage_limits['unit']
        for axis in ['x', 'y', 'z']:
            self.constants.update({axis: self.Q_(self.metadata['constants'][axis])})
            for temp in ['RT', 'LT']:
                lims = [lim *self.ureg(unit) for lim in sorted(self.metadata['voltage_limits'][temp][axis])]
                self.voltage_limits[temp].update({axis: lims})
                
    def _initialize_parameters(self):
        """Add parameters to instrument upon initialization.
        """
        v_limits = []
        for axis in ['x', 'y', 'z']:
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            v_limits += lims_V
        self.add_parameter('position',
                            label='Scanner position',
                            unit='V',
                            vals=vals.Lists(
                                elt_validator=vals.Numbers(min(v_limits), max(v_limits))),
                            get_cmd=self.get_pos,
                            set_cmd=self.goto
                            )
        for i, axis in enumerate(['x', 'y', 'z']):
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            self.add_parameter('position_{}'.format(axis),
                           label='{} position'.format(axis),
                           unit='V',
                           vals=vals.Numbers(min(lims_V), max(lims_V)),
                           get_cmd=(lambda idx=i: self.get_pos()[idx]),
                           set_cmd=getattr(self, '_goto_{}'.format(axis))
                           )
        
    def get_pos(self) -> np.ndarray:
        """Get current scanner [x, y, z] position.

        Returns:
            numpy.ndarray: pos
                Array of current [x, y, z] scanner voltage.
        """
        with nidaqmx.Task('get_pos_ai_task') as ai_task:
            for ax in ['x', 'y', 'z']:
                idx = self.metadata['daq']['channels']['analog_inputs'][ax]
                channel = self.metadata['daq']['name'] + '/ai{}'.format(idx)
                ai_task.ai_channels.add_ai_voltage_chan(channel, ax)
            pos = list(np.round(ai_task.read(), decimals=3))
        for i, ax in enumerate(['x', 'y', 'z']):
            self.metadata['position'].update({ax: '{} V'.format(pos[i])})
        return pos
    
    def goto(self, new_pos: List[float], retract_first: Optional[bool]=False,
             speed: Optional[str]=None, quiet: Optional[bool]=False) -> None:
        """Move scanner to given position.
        By default moves all three axes simultaneously, if necessary.

        Args:
            new_pos: List of [x, y, z] scanner voltage to go to.
            retract_first: If True, scanner retracts to value determined by self.temp,
                then moves in the x,y plane, then moves in z to new_pos. Default: False.
            speed: Speed at which to move the scanner (e.g. '2 V/s') in DAQ voltage units.
                Default set in microscope configuration JSON file.
            quiet: If True, only logs changes in logging.DEBUG mode.
                (goto is called many times during, e.g., a scan.) Default: False.
        """
        old_pos = self.position()
        if speed is None:
            speed = self.speed.to('V/s').magnitude
        else:
            speed = self.Q_(speed).to('V/s').magnitude
        for i, ax in enumerate(['x', 'y', 'z']):
            ax_lim = sorted([lim.to('V').magnitude for lim in self.voltage_limits[self.temp][ax]])
            if new_pos[i] < min(ax_lim) or new_pos[i] > max(ax_lim):
                err = 'Requested position is out of range for {} axis. '
                err += 'Voltage limits are {} V.'
                raise ValueError(err.format(ax, ax_lim))
        if not retract_first:
            ramp = self.make_ramp(old_pos, new_pos, speed)
            with nidaqmx.Task('goto_ao_task') as ao_task:
                for axis in ['x', 'y', 'z']:
                    idx = self.metadata['daq']['channels']['analog_outputs'][axis]
                    channel = self.metadata['daq']['name'] + '/ao{}'.format(idx)
                    ao_task.ao_channels.add_ao_voltage_chan(channel, axis)
                ao_task.timing.cfg_samp_clk_timing(self.daq_rate, samps_per_chan=len(ramp[0]))
                pts = ao_task.write(ramp, auto_start=False)
                ao_task.start()
                ao_task.wait_until_done()
                log.debug('Wrote {} samples to {}.'.format(pts, ao_task.channel_names))
        else:
            self.retract(speed=speed)
            cur_pos = self.get_pos()
            self.goto([new_pos[0], new_pos[1], cur_pos[2]], speed=speed)
            cur_pos = self.get_pos()
            self.goto([cur_pos[0], cur_pos[1], new_pos[2]], speed=speed)
        current_pos = self.position()
        if quiet:
            log.debug('Moved scanner from {} V to {} V.'.format(old_pos, current_pos))
        else:
             log.info('Moved scanner from {} V to {} V.'.format(old_pos, current_pos))
        self.metadata['position'].update({'x': '{} V'.format(current_pos[0]),
                                          'y': '{} V'.format(current_pos[1]),
                                          'z': '{} V'.format(current_pos[2])})
            
    def retract(self, speed: Optional[str]=None) -> None:
        """Retracts z-bender fully based on whether temp is LT or RT.

        Args:
                speed: Speed at which to move the scanner (e.g. '2 V/s') in DAQ voltage units.
                    Default set in microscope configuration JSON file.
        """
        if speed is None:
            speed = self.speed.to('V/s').magnitude
        else:
            speed = self.Q_(speed).to('V/s').magnitude
        current_pos = self.position()
        v_retract = self.Q_(self.voltage_retract[self.temp]).to('V').magnitude
        self.goto([current_pos[0], current_pos[1], v_retract], speed='{} V/s'.format(speed))
    
    def scan_line(self, scan_grids: Dict[str, np.ndarray], ao_channels: Dict[str, int],
                  daq_rate: Union[int, float], counter: Any, reverse=False) -> None:
        """Scan a single line of a plane.

        Args:
            scan_grids: Dict of {axis_name: axis_meshgrid} from utils.make_scan_grids().
            ao_channels: Dict of {axis_name: ao_index} for the scanner ao channels.
            daq_rate: DAQ sampling rate in Hz.
            counter: utils.Counter instance, determines current line of the grid.
            reverse: Determines scan direction (i.e. forward or backward).
        """
        daq_name = self.metadata['daq']['name']
        self.ao_task = nidaqmx.Task('scan_line_ao_task')
        out = []
        line = counter.count
        if reverse:
            step = -1
            last_point = 0
        else:
            step = 1
            last_point = -1
        for axis, idx in ao_channels.items():
            out.append(scan_grids[axis][line][::step])
            self.ao_task.ao_channels.add_ao_voltage_chan('{}/ao{}'.format(daq_name, idx), axis)
        self.ao_task.timing.cfg_samp_clk_timing(daq_rate,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=len(out[0]))
        log.debug('Writing line {}.'.format(line))
        self.ao_task.write(np.array(out), auto_start=False)
        
    def goto_start_of_next_line(self, scan_grids: Dict[str, np.ndarray], counter: Any) -> None:
        """Moves scanner to the start of the next line to scan.

        Args:
            scan_grids: Dict of {axis_name: axis_meshgrid} from utils.make_scan_grids().
            counter: utils.Counter instance, determines current line of the grid.
        """
        line = counter.count
        try:
            start_of_next_line = [scan_grids[axis][line+1][0] for axis in ['x', 'y', 'z']]
            self.goto(start_of_next_line, quiet=True)
        #: If `line` is the last line in the scan, do nothing.
        except IndexError:
            pass

    def check_for_td(self, tdc_plot: Any, data_set: Any, counter: Any) -> None:
        """Check whether touchdown has occurred during a capacitive touchdown.

        Args:
            tdc_plot: plots.TDCPlot instance, which contains current data and parameters
                of the touchdown Loop.
            data_set: DataSet containing capacitance data generated by Loop.
            counter: utils.Counter intance to keep track of which point in the Loop we're at.
        """
        #: If True, will break out of qcodes Loop and and atto.step loop (if applicable)
        self.break_loop = False
        self.td_has_occurred = False
        #: Touchdown z position in DAQ voltage units.
        self.td_height = None
        tdc_plot.update(data_set)
        pt = counter.count
        #: If we've reached the last z position
        if pt >= len(tdc_plot.heights):
            return
        cap_unit = tdc_plot.channels['CAP']['unit']
        #: Initial capacitance should be zero for our capacitance bridges
        initial_cap = self.Q_(tdc_plot.constants['initial_cap']).to(cap_unit).magnitude
        #: Maximum allowed change in capacitance
        max_deltaC = self.Q_(tdc_plot.constants['max_delta_cap']).to(cap_unit).magnitude
        #: Maximum allowed |d(capacitance)/d(voltage)|
        max_slope = self.Q_(tdc_plot.constants['max_slope']).to('{}/V'.format(cap_unit)).magnitude
        prefactor = tdc_plot.prefactors['CAP']
        #: Minimum number of points to fit per line
        nfitmin = tdc_plot.constants['nfitmin']
        #: Width of window used to determine if touchdown has occurred
        #: (should be greater than 2 * nfitmin)
        nwindow = tdc_plot.constants['nwindow']
        #: Minimum number of points to fit per line when determining touchdown point
        #: (should be less than nfitmin)
        ntest = tdc_plot.constants['ntest']
        cdata = tdc_plot.cdata
        hdata = tdc_plot.hdata
        #: Some safety checks:
        if pt > 1:
            if any(abs(cdata[pt-i] - initial_cap) > max_deltaC for i in range(2)):
                log.warning('Capacitance bridge is too unbalanced to continue.')
                self.break_loop = True
                return
            if any(abs(cdata[pt-i] * self.ureg(cap_unit)/prefactor) > self.Q_('5 V') for i in range(2)):
                log.warning('CAP_lockin is railing.')
                self.break_loop = True
                return
        #: Partition data in window into two subsets,
        #: fit a line to each subset, and repeat for next partition
        #: Touchdown point is the partition point that minimizes the sum of squared residuals
        if pt > nwindow:
            #: index of partition boundary corresponding to minimum rms residual
            imin = - nwindow + nfitmin 
            rmsmin = np.inf
            for i in range(-nwindow + nfitmin, -nfitmin):
                p0, rms0 = fit_line(hdata[-nwindow:i+1], cdata[-nwindow:i+1])
                p1, rms1 = fit_line(hdata[i:], cdata[i:])
                rms = rms0 + rms1
                if rms < rmsmin:
                    imin = i
                    rmsmin = rms
            #: Get the slope of the two lines that minimize rms residual
            x0 = hdata[-nwindow:imin+1]
            p0, _ = fit_line(x0, cdata[-nwindow:imin+1])
            x1 = hdata[imin:]
            p1, _ = fit_line(x1, cdata[imin:])
            tdc_plot.ax.plot(x0, p0[0] * x0 + p0[1], 'r-')
            tdc_plot.ax.plot(x1, p1[0] * x1 + p1[1], 'r-')
            tdc_plot.fig.canvas.draw()
            tdc_plot.fig.show()
            if abs(p0[0]) > max_slope:
                log.warning('Pre-touchdown slope +/- {} {}/V is too big.'.format(p0[0], cap_unit))
                self.break_loop = True
                return
            #: If the slopes are different enough, a touchdown has occurred
            if abs(p0[0] - p1[0]) > max_slope:
                self.break_loop = True
                self.td_has_occurred = True
                return

    def get_td_height(self, tdc_plot: Any, task: bool=True) -> None:
        """If a touchdown has occurred, finds the z voltage at which it occurred.

        Args:
            tdc_plot: plots.TDCPlot instance containing data from touchdown.
            task: True if get_td_height is being called as a qcodes Task (no return value allowed).
                Default True.
        """
        if self.td_has_occurred or not task:
            cdata = tdc_plot.cdata
            hdata = tdc_plot.hdata
            cap_unit = tdc_plot.channels['CAP']['unit']
            nwindow = tdc_plot.constants['nwindow']
            ntest = tdc_plot.constants['ntest']
            #: Partition data in window into two subsets,
            #: fit a line to each subset, and repeat for next partition.
            #: Touchdown point is the partition point that minimizes the sum of squared residuals
            imin = -ntest
            rmsmin = np.inf
            for i in range(-nwindow + ntest, -ntest):
                p0, rms0 = fit_line(hdata[-nwindow:i+1], cdata[-nwindow:i+1])
                p1, rms1 = fit_line(hdata[i:], cdata[i:])
                rms = rms0 + rms1
                if rms < rmsmin:
                    imin = i
                    rmsmin = rms
            #: Get the slope of the two lines that minimize rms residual
            p0, _ = fit_line(hdata[-nwindow:imin+1], cdata[-nwindow:imin+1])
            p1, _ = fit_line(hdata[imin:], cdata[imin:])
            self.td_height = (p1[1]-p0[1]) / (p0[0] - p1[0])
            tdc_plot.td_height = self.td_height
            tdc_plot.pre_td_slope = '{} {}/V'.format(p0[0], cap_unit)
            tdc_plot.post_td_slope = '{} {}/V'.format(p1[0], cap_unit)
            tdc_plot._clear_artists()
            tdc_plot.ax.plot(hdata, cdata, 'b.')
            tdc_plot.ax.plot(hdata[-1], cdata[-1], 'r.')
            tdc_plot.ax.plot(hdata[-nwindow:i+3], hdata[-nwindow:i+3] * p0[0] + p0[1], 'r-')
            tdc_plot.ax.plot(hdata[i-2:], hdata[i-2:] * p1[0] + p1[1], 'r-')
            tdc_plot.ax.set_title('Touchdown: {:.4} V'.format(self.td_height))
            tdc_plot.fig.canvas.draw()
            tdc_plot.fig.show()
            log.info('Touchdown occurred at {:.4} V.'.format(self.td_height))
            if self.td_height < 0:
                msg = 'Touchdown occurred at a negative voltage. '
                msg += 'Consider Atto stepping further from the sample '
                msg += 'to touchdown at a positive voltage.'
                log.warning(msg)
            log.info('Pre-touchdown slope: {} {}/V.'.format(tdc_plot.pre_td_slope, cap_unit))
            log.info('Post-touchdown slope: {} {}/V.'.format(tdc_plot.post_td_slope, cap_unit))
            self.metadata['position']['plane'].update({'z': self.td_height})
        if not task:
            return self.td_height
    
    def clear_instances(self):
        """Clear scanner instances.
        """
        for inst in self.instances():
            self.remove_instance(inst)
            
    def control_ao_task(self, cmd: str) -> None:
        """Write commands to the DAQ AO Task. Used during qc.Loops.

        Args:
            cmd: What you want the Task to do. For example,
                self.control_ao_task('stop') is equivalent to self.ao_task.stop()
        """
        if hasattr(self, 'ao_task'):
            getattr(self.ao_task, cmd)()

    def make_ramp(self, pos0: List, pos1: List, speed: Union[int, float]) -> np.ndarray:
        """Generates a ramp in x,y,z scanner voltage from point pos0 to point pos1 at given speed.

        Args:
            pos0: List of initial [x, y, z] scanner voltages.
            pos1: List of final [x, y, z] scanner votlages.
            speed: Speed at which to go to pos0 to pos1, in DAQ voltage/second.

        Returns:
            numpy.ndarray: ramp
                Array of x, y, z values to write to DAQ AOs to move
                scanner from pos0 to pos1.
        """
        if speed > self.speed.to('V/s').magnitude:
            msg = 'Setting ramp speed to maximum allowed: {} V/s.'
            log.warning(msg.format(self.speed.to('V/s').magnitude))
        pos0 = np.array(pos0)
        pos1 = np.array(pos1)
        max_ramp_distance = np.max(np.abs(pos1-pos0))
        ramp_time = max_ramp_distance/speed
        npts = int(ramp_time * self.daq_rate) + 2
        ramp = []
        for i in range(3):
            ramp.append(np.linspace(pos0[i], pos1[i], npts))
        return np.array(ramp)
    
    def _goto_x(self, xpos: float) -> None:
        """Go to give x position.

        Args:
            xpos: x position to go to, in DAQ voltage.
        """
        current_pos = self.get_pos() 
        self.goto([xpos, current_pos[1], current_pos[2]], quiet=True)
        
    def _goto_y(self, ypos: float) -> None:
        """Go to give y position.

        Args:
            ypos: y position to go to, in DAQ voltage.
        """
        current_pos = self.position()
        self.goto([current_pos[0], ypos, current_pos[2]], quiet=True)
    
    def _goto_z(self, zpos: float) -> None:
        """Go to give z position.

        Args:
            zpos: z position to go to, in DAQ voltage.
        """
        current_pos = self.position()
        self.goto([current_pos[0], current_pos[1], zpos], quiet=True)
