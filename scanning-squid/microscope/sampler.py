#: Various Python utilities
from typing import Dict, List, Sequence, Any, Union, Tuple
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from scipy import io

#: Qcodes for running measurements and saving data
import qcodes as qc
from qcoces.data.io import DiskIO

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from instruments.dg645 import DG645
from instruments.afg3000 import AFG3252
from .micropscope import Microscope
from utils import Counter, clear_artists

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
        super().__init__(config_file, temp, ureg, log_level, log_name, **kwargs)
        self._add_delay_generator()
        self._add_function_generator()

    def _add_delay_generator(self):
        """Add SRS DG645 digital delay generator to SamplerMicroscope.
        """
        info = self.config['instruments']['delay_generator']
        name = 'dg'
        if hasattr(self, name):
            getattr(self, name, 'clear_instances')()
        self.remove_component(name)
        instr = DG645(name, info['address'], metadata={info})
        setattr(self, name, instr)
        self.add_component(getattr(self, name))
        log.info('{} successfully added to microscope.'.format(name))

    def _add_function_generator(self):
        """Add Tektronix AFG3000 series function generator to SamplerMicroscope.
        """
        info = self.config['instruments']['function_generator']
        name = 'afg'
        if hasattr(self, name):
            getattr(self, name, 'clear_instances')()
        self.remove_component(name)
        instr = AFG3252(name, info['address'], metadata={info})
        setattr(self, name, instr)
        self.add_component(getattr(self, name))
        log.info('{} successfully added to microscope.'.format(name))

    # def get_prefactors(self, measurement: Dict[str, Any], update: bool=True) -> Dict[str, Any]:
    #     """For each channel, calculate prefactors to convert DAQ voltage into real units.

    #     Args:
    #         measurement: Dict of measurement parameters as defined
    #             in measurement configuration file.
    #         update: Whether to query instrument parameters or simply trust the
    #             latest values (should this even be an option)?

    #     Returns:
    #         Dict[str, pint.Quantity]: prefactors
    #             Dict of {channel_name: prefactor} where prefactor is a pint Quantity.
    #     """
    #     prefactors = {}
    #     for ch in measurement['channels']:
    #         prefactor = 1
    #         if ch == 'MAG':
    #             snap = getattr(self, 'MAG_lockin').snapshot(update=update)['parameters']
    #             mag_sensitivity = snap['sensitivity']['value']
    #             #amp = snap['amplitude']['value'] * self.ureg(snap['amplitude']['unit'])
    #             #: The factor of 10 here is because SR830 output gain is 10/sensitivity
    #             prefactor /=  self.Q_(10 / mag_sensitivity)
    #         elif ch == 'CAP':
    #             snap = getattr(self, 'CAP_lockin').snapshot(update=update)['parameters']
    #             cap_sensitivity = snap['sensitivity']['value']
    #             #: The factor of 10 here is because SR830 output gain is 10/sensitivity
    #             prefactor /= (self.Q_(self.scanner.metadata['cantilever']['calibration']) * 10 / cap_sensitivity)
    #         prefactor /= measurement['channels'][ch]['gain']
    #         prefactors.update({ch: prefactor})
    #     return prefactors

    def iv_mod(self, ivm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Measures IV characteristic at different mod coil voltages.

        Args:
            ivm_params: Dict of measurement parameters as definted in config_measurements json file.

        Returns:

            Dict: out_dict
                Dictionary containing data arrays and instrument metadata.
        """

        out_dict = {}
        mod_range = [self.Q_(value).to('V').magnitude for value in ivm_params['mod_range']]
        bias_range = [self.Q_(value).to('V').magnitude for value in ivm_params['bias_range']]
        mod_vec = np.linspace(mod_range[0], mod_range[1], ivm_params['ntek2'])
        bias_vec = np.linspace(bias_range[0], bias_range[1], ivm_params['ntek1'])
        mod_grid, bias_grid  = np.meshgrid(mod_vec, bias_vec)
        ivmX = np.full_like(mod_vec, np.nan, dtype=np.double)
        yvmY = np.full_like(mod_vec, np.nan, dtype=np.double)
        
        #: Set AFG output channels
        self.afg.voltage_low1('0V')
        self.afg.voltage_high1('{}V'.format(bias_range[0]))

        self.afg.voltage_low2('0V')
        self.afg.voltage_high2('{}V'.format(mod_range[0]))
        self.afg.voltage_offset2('0V')

        for ch in [1, 2]:
            p = ivm_params['afg']['ch{}'.format(ch)]
            getattr(self.afg, 'pulse_period{}'.format(ch))('{}us'.format(self.Q_(p['period']).to('us').magnitude))
            getattr(self.afg, 'pulse_width{}'.format(ch))('{}us'.format(self.Q_(p['width']).to('us').magnitude))
            getattr(self.afg, 'pulse_trans_lead{}'.format(ch))('{}us'.format(self.Q_(p['lead']).to('us').magnitude))
            getattr(self.afg, 'pulse_trans_trail{}'.format(ch))('{}us'.format(self.Q_(p['trail']).to('us').magnitude))
            getattr(self.afg, 'pulse_delay{}'.format(ch))('{}us'.format(self.Q_(p['delay']).to('us').magnitude))

        lockin_snap = self.MAG_lockin.snapshot(update=True)
        out_dict.update({'metadata':
                            {'lockin': lockin_snap,
                             'afg': self.afg.snapshot(update=True),
                             'ivm_params': ivm_params}
                        })
        prefactor = 1 / (10 / lockin_snap['parameters']['sensitivity']['value'])
        prefactor /= ivm_params['channels']['ivmX']['gain']
        delay = ivm_params['wait_factor'] * lockin_snap['parameters']['time_constant']['value']

        fig, ax = plt.subplots(1)
        ax.set_xlim(min(bias_range), max(bias_range))
        ax.set_xlabel('Bias [V]')
        ax.set_ylabel('Voltage [V]')
        ax.set_title(ivm_params['channels']['X']['label'])

        for j in range(len(mod_vec)):
            dataX, dataY, dataX_avg, dataY_avg = (np.zeros(len(mod_vec)), ) * 4
            self.afg.voltage_offset2(mod_vec[j])
            for _ in range(ivm_params['navg']):
                for i in range(len(bias_vec)):
                    self.afg.voltage_high1(bias_vec[i])
                    time.sleep(delay)
                    dataX[i] = self.MAG_lockin.X()
                    dataY[i] = self.MAG_lockin.Y()
                dataX_avg += dataX
                dataY_avg += dataY
            dataX_avg /= ivm_params['navg']
            dataY_avg /= ivm_params['navg']
            ivmX[:,j] = prefactor * dataX_avg
            ivmY[:,j] = prefactor * dataY_avg
            clear_artists(ax)
            ax.plot(bias_vec, dataX_avg, 'bo-', label='X')
            fig.canvas.draw()
        fig.show()
        
        fig1 = plt.figure(1)    
        plt.pcolormesh(mod_grid, bias_grid, imvX)
        plt.xlabel('Modulation [V]')
        plt.ylabel('Bias [V]')
        plt.title(ivm_params['channel']['X']['label'])
        cbar1 = plt.colorbar()
        cbar1.set_label('Voltage [V]')

        fig2 = plt.figure(2)    
        plt.pcolormesh(mod_grid, bias_grid, imvY)
        plt.xlabel('Modulation [V]')
        plt.ylabel('Bias [V]')
        plt.title(ivm_params['channel']['Y']['label'])
        cbar2 = plt.colorbar()
        cbar2.set_label('Voltage [V]')

        out_dict.update(
            {'data':
                {'ivmX': {'array': ivmX, 'unit': 'V'},
                 'ivmY': {'array': ivmY, 'unit': 'V'}}
            }
        )

        if ivm_params['save']:
            loc_provider = FormatLocation(fmt='{date}/#{counter}_{name}_{time}')
            loc = loc_provider(DiskIO('.'), record={'name': ivm_params['fname']})
            io.savemat('{}/{}'.format(loc, ivm_params['fname']), out_dict)
            fig1.savefig('{}/{}X.png'.format(loc, ivm_params['fname']))
            fig2.savefig('{}/{}Y.png'.format(loc, ivm_params['fname']))

        return out_dict


