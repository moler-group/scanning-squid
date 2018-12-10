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

from qcodes.instrument.visa import VisaInstrument
from typing import List, Dict, Union, Optional, Sequence, Any, Tuple
import qcodes.utils.validators as vals
import numpy as np
import visa
import time

class EL320P(VisaInstrument):
    """Qcodes driver for AIM & Thurlby Thandar EL320P power supply.
    Logan Bishop-Van Horn (2018)
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\r\n', **kwargs)
        self.visa_handle.baud_rate = 9600
        self.add_parameter(
            name='voltage_set',
            label='Voltage setting',
            unit='V',
            get_cmd='V?',
            get_parser=self._get_parser,
            set_cmd='V {}',
            vals=vals.Numbers(0,30)
            )
        self.add_parameter(
            name='voltage_out',
            label='Actual output voltage',
            unit='V',
            get_cmd='VO?',
            get_parser=self._get_parser
            )
        self.add_parameter(
            name='current_set',
            label='Current setting',
            unit='A',
            get_cmd='I?',
            get_parser=self._get_parser,
            set_cmd='I {}',
            vals=vals.Numbers(0.01,2)
            )
        self.add_parameter(
            name='current_out',
            label='Actual output current',
            unit='A',
            get_cmd='IO?',
            get_parser=self._get_parser
            )
        self.add_parameter(
            name='mode',
            label='Mode',
            unit='',
            get_cmd='M?',
            get_parser=str
            )
        self.add_parameter(
            name='output',
            label='Output status',
            unit='',
            get_cmd='OUT?',
            get_parser=self._output_parser,
            set_cmd='{}',
            vals=vals.Enum('ON', 'OFF')
            )
        self.add_parameter(
            name='error',
            label='Error message',
            unit='',
            get_cmd='ERR?',
            get_parser=self._error_parser,
            val_mapping={'OK': 0, 'Command not recognized': 1, 'Value outside of instrument limits': 2},
            vals=vals.Enum(0,1,2)
            )
        self.connect_message()

    def _get_parser(self, response):
        return float(response[2:])

    def _output_parser(self, response):
        return response[4:]

    def _error_parser(self, response):
        return int(response[-1])