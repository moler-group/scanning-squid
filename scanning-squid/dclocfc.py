# Copied scanner.py and edited for 1 channel Analog Output for dc local field (Yusuke Iguchi, 1/3/2022)
#
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

import qcodes as qc
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
import utils
from scipy import io
from scipy.interpolate import Rbf
from typing import Dict, List, Optional, Sequence, Any, Union
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType, TaskMode
import logging
log = logging.getLogger(__name__)

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

class Dclocfc(Instrument):
    """Controls DAQ AOs to drive the dclocalfield.
    """   
    def __init__(self, dclocfc_config: Dict[str, Any], daq_config: Dict[str, Any],
                 temp: str, ureg: Any, **kwargs) -> None:
        """
        Args:
            dclocfc_config: dc local field configuration dictionary as defined
                in microscope configuration JSON file.
            daq_config: DAQ configuration dictionary as defined
                in microscope configuration JSON file.
            temp: 'LT' or 'RT' - sets the dc local field voltage limit
                based on temperature mode.
            ureg: pint UnitRegistry, manages units.
        """
        super().__init__(dclocfc_config['name'], **kwargs)
        if temp.upper() not in ['LT', 'RT']:
            raise ValueError('Temperature mode must be "LT" or "RT".')
        self.temp = temp.upper()
        self.ureg = ureg
        self.Q_ = ureg.Quantity
        self.metadata.update(dclocfc_config)
        self.metadata.update({'daq': daq_config})
        self._parse_unitful_quantities()
        self._initialize_parameters()
        self.goto([0])
        
    def _parse_unitful_quantities(self):
        """Parse strings from configuration dicts into Quantities with units.
        """
        self.daq_rate = self.Q_(self.metadata['daq']['rate']).to('Hz').magnitude
        self.speed = self.Q_(self.metadata['speed']['value'])
        self.resistance = self.Q_(self.metadata['resistance']['value']).to('Ohm').magnitude
        self.voltage_limits = {'RT': {},
                               'LT': {},
                               'unit': self.metadata['voltage_limits']['unit'],
                               'comment': self.metadata['voltage_limits']['comment']}
        unit = self.voltage_limits['unit']
        for axis in ['dcFC']:
            for temp in ['RT', 'LT']:
                lims = [lim *self.ureg(unit) for lim in sorted(self.metadata['voltage_limits'][temp][axis])]
                self.voltage_limits[temp].update({axis: lims})
                
    def _initialize_parameters(self):
        """Add parameters to instrument upon initialization.
        """
        v_limits = []
        for axis in ['dcFC']:
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            v_limits += lims_V
        self.add_parameter('voltage',
                            label='dc local field voltage',
                            unit='V',
                            vals=vals.Lists(
                                elt_validator=vals.Numbers(min(v_limits), max(v_limits))),
                            get_cmd=self.get_pos,
                            set_cmd=self.goto
                            )
        for i, axis in enumerate(['dcFC']):
            lims = self.voltage_limits[self.temp][axis]
            lims_V = [lim.to('V').magnitude for lim in lims]
            self.add_parameter('voltage_{}'.format(axis),
                           label='{} voltage'.format(axis),
                           unit='V',
                           vals=vals.Numbers(min(lims_V), max(lims_V)),
                           get_cmd=(lambda idx=i: self.get_pos()[idx]),
                           set_cmd=getattr(self, '_goto_{}'.format(axis))
                           )
        
    def get_pos(self) -> np.ndarray:
        """Get current dc local field voltage.

        Returns:
            numpy.ndarray: pos
                Array of current [dcFC] dc local field voltage.
        """    
        with nidaqmx.Task('get_pos_ai_task') as ai_task:
            for ax in ['dcFC']:
                idx = self.metadata['daq']['channels']['analog_inputs'][ax]
                channel = self.metadata['daq']['name'] + '/ai{}'.format(idx)
                ai_task.ai_channels.add_ai_voltage_chan(channel, ax, min_val=-10, max_val=10)
            pos_raw = list([np.round(ai_task.read(), decimals=3)])
        pos = []
        for i, ax in enumerate(['dcFC']):
            ax_lim = sorted([lim.to('V').magnitude for lim in self.voltage_limits[self.temp][ax]])
            if pos_raw[i] < ax_lim[0]:
                pos.append(ax_lim[0])
            elif pos_raw[i] > ax_lim[1]:
                pos.append(ax_lim[1])
            else:
                pos.append(pos_raw[i])
        return pos
    
    def goto(self, new_pos: List[float],
             speed: Optional[str]=None, quiet: Optional[bool]=False) -> None:
        """Move scanner to given position.
        By default moves all three axes simultaneously, if necessary.

        Args:
            new_pos: List of [dcFC] dc local field voltage to go to.
            speed: Speed at which to change the dc local field voltage (e.g. '2 V/s') in DAQ voltage units.
                Default set in microscope configuration JSON file.
            quiet: If True, only logs changes in logging.DEBUG mode.
                (goto is called many times during, e.g., a scan.) Default: False.
        """
        old_pos = self.voltage()
        if speed is None:
            speed = self.speed.to('V/s').magnitude
        else:
            speed = self.Q_(speed).to('V/s').magnitude
        for i, ax in enumerate(['dcFC']):
            ax_lim = sorted([lim.to('V').magnitude for lim in self.voltage_limits[self.temp][ax]])
            if new_pos[i] < min(ax_lim) or new_pos[i] > max(ax_lim):
                err = 'Requested position is out of range for {} axis. '
                err += 'Voltage limits are {} V.'
                raise ValueError(err.format(ax, ax_lim))
        
        ramp = self.make_ramp(old_pos, new_pos, speed)
        with nidaqmx.Task('goto_ao_task') as ao_task:
            for axis in ['dcFC']:
                idx = self.metadata['daq']['channels']['analog_outputs'][axis]
                channel = self.metadata['daq']['name'] + '/ao{}'.format(idx)
                ao_task.ao_channels.add_ao_voltage_chan(channel, axis)
            ao_task.timing.cfg_samp_clk_timing(self.daq_rate, samps_per_chan=len(ramp))
            # log.info('daq_rate: {}, samps_per_chan={}. data={}'.format(self.daq_rate,len(ramp[0]),ramp))
            pts = ao_task.write(ramp, auto_start=False, timeout=30)
            ao_task.start()
            ao_task.wait_until_done()
            log.debug('Wrote {} samples to {}.'.format(pts, ao_task.channel_names))
        current_pos = self.voltage()
        if quiet:
            log.debug('Change dc local field voltage from {} V to {} V with {} Ohm resistor.'.format(old_pos, current_pos, self.resistance))
        else:
             log.info('Change dc local field voltage from {} V to {} V with {} Ohm resistor.'.format(old_pos, current_pos, self.resistance))
            
            

    def clear_instances(self):
        """Clear dclocfc instances.
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
            pos0: List of initial [dcFC] voltage.
            pos1: List of final [dcFC] votlage.
            speed: Speed at which to go to pos0 to pos1, in DAQ voltage/second.

        Returns:
            numpy.ndarray: ramp
                Array of [dcFC] values to write to DAQ AOs to change dc local field
                voltage from pos0 to pos1.
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
        for i in range(1):
            ramp.append(np.linspace(pos0[i], pos1[i], npts))
        # np.squeeze() changes 2D array to 1D array if ramp has only one channel
        return np.squeeze(np.array(ramp))
        
    
    def _goto_dcFC(self, xpos: float) -> None:
        """Go to given dcFC voltage.

        Args:
            xpos: dcFC voltage to go to, in DAQ voltage.
        """
        self.goto([xpos], quiet=True)
        
