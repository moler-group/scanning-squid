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

from qcodes import VisaInstrument
import qcodes.utils.validators as vals
import logging
log = logging.getLogger(__name__)

class DG645(VisaInstrument):
    """Qcodes driver for SRS DG645 digital delay generator.

    Not all instrument functionality is included here.
    Logan Bishop-Van Horn (2018)
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', timeout=10, **kwargs)
        self.channel_mapping = {
            'T0': 0, 'T1': 1, 'A': 2, 'B': 3, 'C': 4,
            'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9
        }
        self.output_mapping = {'T0': 0, 'AB': 1, 'CD': 2, 'EF': 3, 'GH': 4}
        # self.display_mapping = {
        #     'trig_rate': 0,
        #     'trig_thresh': 1,
        #     'trig_single_shot': 2,
        #     'trig_line': 3,
        #     'advanced_trig_enable': 4,
        #     'trig_holdoff': 5,
        #     'prescale_config': 6,
        #     'burst_mode': 7,
        #     'burst_delay': 8,
        #     'burst_count': 9,
        #     'burst_period': 10,
        #     'channel_delay': 11,
        #     'channel_output_levels': 12,
        #     'channel_output_polarity': 13,
        #     'burst_T0_config': 14
        # }
        self.prescale_mapping = {'trig': 0, 'AB': 1, 'CD': 2, 'EF': 3, 'GH': 4}
        self.trig_mapping = {
                'internal': 0,
                'ext_rising': 1,
                'ext_falling': 2,
                'single_ext_rising': 3,
                'single_ext_falling': 4,
                'single': 5,
                'line': 6
        } 
        self.polarity_mapping = {'-': 0, '+': 1}

        self.add_parameter('trig_holdoff',
               label='Trigger holdoff',
               unit='s',
               get_cmd='HOLD?',
               get_parser=float,
               set_cmd='HOLD {}'
        )

        #: Prescale parameters
        for k, v in self.prescale_mapping.items():
            if v > 0:
                self.add_parameter('phase_{}'.format(k),
                       label='Prescale phase factor {}'.format(k),
                       unit='',
                       get_cmd=lambda ch=k: self._get_phase_prescale(ch),
                       get_parser=int,
                       set_cmd=lambda val, ch=k: self._set_phase_prescale(val, channel=ch),
                       vals=vals.Ints(min_value=0)
                )

            self.add_parameter('prescale_{}'.format(v),
                   label='Prescale factor {}'.format(v),
                   unit='',
                   get_cmd=lambda ch=k: self._get_prescale(ch),
                   get_parser=int,
                   set_cmd=lambda val, ch=k: self._set_prescale(val, channel=ch),
                   vals=vals.Ints(min_value=0)
            )

        #: Trigger parameters
        self.add_parameter('trig_level',
               label='Trigger level',
               unit='V',
               get_cmd='TLVL?',
               get_parser=float,
               set_cmd='TLVL {}',
               vals=vals.Numbers()
        )
        self.add_parameter('trig_rate',
               label='Trigger rate',
               unit='Hz',
               get_cmd='TRAT?',
               get_parser=float,
               set_cmd='TRAT {}',
               vals=vals.Numbers(min_value=0)
        ) 
        self.add_parameter('trig_source',
               label='Trigger source',
               unit='',
               get_cmd=self._get_trig_source,
               get_parser=str,
               set_cmd=self._set_trig_source,
               vals=vals.Enum(tuple(self.trig_mapping.keys()))
        )

        #: Burst parameters
        self.add_parameter('burst_count',
               label='Burst count',
               unit='',
               get_cmd='BURC?',
               get_parser=int,
               set_cmd='BURC {}',
               vals=vals.Ints(min_value=0)
        )
        self.add_parameter('burst_delay',
               label='Burst delay',
               unit='s',
               get_cmd='BURD?',
               get_parser=float,
               set_cmd='BURD {}',
               vals=vals.Numbers(min_value=0)
        )
        self.add_parameter('burst_period',
               label='Burst period',
               unit='s',
               get_cmd='BURP?',
               get_parser=float,
               set_cmd='BURC {}',
               vals=vals.Numbers(min_value=100e-9, max_value=2000-10e-9)
        )
        self.add_parameter('burst_T0_config',
               label='Burst T0 configuration',
               unit='',
               get_cmd='BURT?',
               get_parser=int,
               set_cmd='BURT {}',
               vals=vals.Enum(0,1)
        )

        #: Channel parameters
        for ch, idx in self.channel_mapping.items():
            if idx > 1:
                self.add_parameter('delay_{}'.format(ch),
                       label='{} delay'.format(ch),
                       unit='s',
                       get_cmd=lambda c=ch: self._get_delay(channel=c),
                       get_parser=str,
                       set_cmd=lambda src_delay, c=ch: self._set_delay(src_delay, target=c),
                       vals=vals.Strings()
                )
                self.add_parameter('channel_link_{}'.format(ch),
                       label='Channel linked to {}'.format(ch),
                       unit='',
                       get_cmd=lambda c=ch: self._get_link(channel=c),
                       get_parser=int,
                       set_cmd=lambda d, c=ch: self._set_link(d, channel=c),
                       vals=vals.Enum(tuple(k for k in self.channel_mapping if k != 'T1'))
                )

        #: Output parameters
        for out, idx in self.output_mapping.items():
            self.add_parameter('amp_out_{}'.format(out),
                   label='Output {} amplitude'.format(out),
                   unit='V',
                   get_cmd=lambda o=out: self._get_amp(output=o),
                   get_parser=float,
                   set_cmd=lambda lvl, o=out: self._set_amp(lvl, output=o),
                   vals=vals.Numbers()
            )
            self.add_parameter('offset_out_{}'.format(out),
                   label='Output {} offset'.format(out),
                   unit='V',
                   get_cmd=lambda o=out: self._get_offset(output=o),
                   get_parser=float,
                   set_cmd=lambda lvl, o=out: self._set_offset(lvl, output=o),
                   vals=vals.Numbers()
            )
            self.add_parameter('polarity_out_{}'.format(out),
                   label='Output {} polarity'.format(out),
                   unit='',
                   get_cmd=lambda o=out: self._get_polarity(output=o),
                   get_parser=int,
                   set_cmd=lambda lvl, o=out: self._set_offset(lvl, output=o),
                   vals=vals.Enum(0,1)
            )

        self.snapshot(update=True)
        self.connect_message()

    def calibrate(self) -> None:
        """Run auto-calibration routine.
        """
        self.write('*CAL?')
        self.wait()

    def self_test(self) -> None:
        """Run self-test routine.
        """
        self.write('*TST?')
        self.wait()

    def reset(self) -> None:
        """Reset instrument.
        """
        log.info('Resetting {}.'.format(self.name))
        self.write('*RST')

    def save_settings(self, location: int) -> None:
        """Save instrument settings to given location.

        Args:
            location: Location to which to save the settings (in [1..9]).
        """
        log.info('Saving instrument settings to location {}.'.format(location))
        self.write('*SAV {}'.format(location))

    def trigger(self) -> None:
        """Initiates a single trigger if instrument is in single shot mode.
        """
        self.write('*TRG')

    def wait(self) -> None:
        """Wait for all prior commands to execute before continuing.
        """
        self.write('*WAI')

    def local(self) -> None:
        """Go to local.
        """
        self.write('LCAL')

    def remote(self) -> None:
        """Go to remote.
        """
        self.write('REMT')

    def _get_phase_prescale(self, channel: str) -> str:
        return self.ask('PHAS?{}'.format(self.prescale_mapping[channel]))

    def _set_phase_prescale(self, value: int, channel: str=None) -> None:
        self.write('PHAS {},{}'.format(self.prescale_mapping[channel], value))

    def _get_prescale(self, channel: str) -> str:
        return self.ask('PRES?{}'.format(self.prescale_mapping[channel]))

    def _set_prescale(self, value: int, channel: str=None) -> None:
        self.write('PRES {},{}'.format(self.prescale_mapping[channel], value))

    def _set_trig_source(self, src: str) -> None:  
        self.write('TSRC {}'.format(self.trig_mapping[src]))

    def _get_trig_source(self) -> str:
        response = self.ask('TSRC?')
        keys = self.trig_mapping.keys()
        values = self.trig_mapping.values()
        return list(keys)[list(values).index(int(response))]

    def _get_delay(self, channel: str=None) -> str:
        return self.ask('DLAY?{}'.format(self.channel_mapping[channel]))

    def _set_delay(self, src_delay: str, target: str=None) -> None:
        source, delay = [s.strip() for s in src_delay.split(',')]
        self.write('DLAY {},{},{}'.format(self.channel_mapping[target],
                                          self.channel_mapping[source],
                                          delay))

    def _get_amp(self, output: str=None) -> str:
        return self.ask('LAMP?{}'.format(self.output_mapping[output]))

    def _set_amp(self, lvl: float, output: str=None) -> None:
        self.write('LAMP {},{}'.format(lvl, self.output_mapping[output]))

    def _get_link(self, channel: str=None) -> str:
        return self.ask('LINK?{}'.format(self.channel_mapping[channel]))

    def _set_link(self, target: str, source: str=None) -> None:
        self.write('LINK {},{}'.format(self.channel_mapping[source],
                                       self.channel_mapping[target]))

    def _get_offset(self, output: str=None) -> str:
        return self.ask('LOFF?{}'.format(self.output_mapping[output]))

    def _set_offset(self, off: float, output: str=None) -> None:
        self.write('LOFF {},{}'.format(off, self.output_mapping[output]))

    def _get_polarity(self, output: str=None) -> str:
        return self.ask('LPOL?{}'.format(self.output_mapping[output]))

    def _set_polarity(self, pol: int, output: str=None) -> None:
        self.write('LPOL {},{}'.format(pol, self.output_mapping[output]))

    def clear_instances(self):
        """Clear instances of DG645 Instruments.
        """
        for instance in self.instances():
            self.remove_instance(instance)