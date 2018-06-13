import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
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
        
    def _parse_unitful_quantities(self):
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
        """
        Add parameters to instrument upon initialization.
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
        #for axis, idx in self.metadata['daq']['channels']['analog_outputs'].items():
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
        
    def get_pos(self) -> List[float]:
        """Get current scanner [x, y, z] position.
        """
        with nidaqmx.Task('get_pos_ai_task') as ai_task:
            for ax in ['x', 'y', 'z']:
                idx = self.metadata['daq']['channels']['analog_inputs'][ax]
                channel = self.metadata['daq']['name'] + '/ai{}'.format(idx)
                ai_task.ai_channels.add_ai_voltage_chan(channel, ax)
            pos = list(ai_task.read())
        for i, ax in enumerate(['x', 'y', 'z']):
            self.metadata['position'].update({ax: '{} V'.format(pos[i])})
        return pos
    
    def goto(self,
             new_pos: str,
             retract_first: Optional[bool]=False,
             speed: Optional[str]=None,
             quiet: Optional[bool]=False) -> None:
        """
        Move scanner to given position.
        By default moves all three axes simultaneously, if necessary.
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
            ramp = self._make_ramp(old_pos, new_pos, speed)
            with nidaqmx.Task('goto_ao_task') as ao_task:
                for axis in ['x', 'y', 'z']:
                    idx = self.metadata['daq']['channels']['analog_outputs'][axis]
                    channel = self.metadata['daq']['name'] + '/ao{}'.format(idx)
                    ao_task.ao_channels.add_ao_voltage_chan(channel, axis)
                ao_task.timing.cfg_samp_clk_timing(self.daq_rate, samps_per_chan=len(ramp[0]))
                pts = ao_task.write(ramp, auto_start=False)
                ao_task.start()
                ao_task.wait_until_done()
                self.metadata['position'].update({'x': '{} V'.format(new_pos[0]),
                                                  'y': '{} V'.format(new_pos[1]),
                                                  'z': '{} V'.format(new_pos[2])})
                log.debug('Wrote {} samples to {}.'.format(pts, ao_task.channel_names))
        else:
            self.retract(speed=speed)
            cur_pos = self.get_pos()
            self.goto([new_pos[0], new_pos[1], cur_pos[2]], speed=speed)
            cur_pos = self.get_pos()
            self.goto([cur_pos[0], cur_pos[1], new_pos[2]], speed=speed)
        if quiet:
            log.debug('Moved scanner from {} V to {} V.'.format(old_pos, new_pos))
        else:
             log.info('Moved scanner from {} V to {} V.'.format(old_pos, new_pos))
            
    def retract(self, speed: Optional[str]=None) -> None:
        """
        Retracts z-bender fully based on whether temp is LT or RT.
        """
        if speed is None:
            speed = self.speed.to('V/s').magnitude
        else:
            speed = self.Q_(speed).to('V/s').magnitude
        old_pos = self.position()
        v_retract = self.Q_(self.voltage_retract[self.temp]).to('V').magnitude
        self.goto([old_pos[0], old_pos[1], v_retract], speed='{} V/s'.format(speed))
    
    def scan_line(self,
                  scan_grids: Dict[str, Any],
                  ao_channels: Dict[str, int],
                  daq_rate: Union[int, float],
                  counter: Any,
                  reverse=False) -> None:
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
        for axis in ['x', 'y', 'z']:
            self.metadata['position'].update({
                axis: '{} V'.format(scan_grids[axis][line][last_point])
            })
        
    def goto_start_of_next_line(self, scan_grids, counter):
        line = counter.count
        try:
            start_of_next_line = [scan_grids[axis][line+1][0] for axis in ['x', 'y', 'z']]
            self.goto(start_of_next_line, quiet=True)
        except IndexError:
            pass

    def check_for_td(self, loop, tdc_params):
        self.scanner.td_has_occurred = False
    
    def clear_instances(self):
        for inst in self.instances():
            self.remove_instance(inst)
            
    def control_ao_task(self, cmd):
        if hasattr(self, 'ao_task'):
            getattr(self.ao_task, cmd)()

    def _make_ramp(self, pos0: List, pos1: List, speed: Union[int, float]) -> np.ndarray:
        """
        Generates a ramp in x,y,z scanner voltage from
        point pos0 to point pos1 at given speed.
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
    
    def _goto_x(self, xpos: List[float]):
        pos = self.get_pos() 
        self.goto([xpos, pos[1], pos[2]], quiet=True)
        
    def _goto_y(self, ypos: List[float]):
        pos = self.get_pos()
        self.goto([pos[0], ypos, pos[2]], quiet=True)
    
    def _goto_z(self, zpos: List[float]):
        pos = self.get_pos()
        self.goto([pos[0], pos[1], zpos], quiet=True)