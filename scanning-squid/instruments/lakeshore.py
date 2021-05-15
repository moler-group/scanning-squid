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

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum, Strings, Numbers
import visa

class SensorChannel(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """

    _CHANNEL_VAL = Enum("A", "B")

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        # Validate the channel value
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel  # Channel on the temperature controller. Can be A-B

        # Add the various channel parameters
        self.add_parameter('temperature', get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temerature',
                           unit='K')
        self.add_parameter('sensor_raw', get_cmd='SRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Raw_Reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type
        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel),
                           val_mapping={'OK': 0, 'Invalid Reading': 1, 'Temp Underrange': 16, 'Temp Overrange': 32,
                           'Sensor Units Zero': 64, 'Sensor Units Overrange': 128}, label='Sensor_Status')
        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=Strings(15),
                           label='Sensor_Name')


class Model_335(VisaInstrument):
    """
    Lakeshore Model 335 Temperature Controller Driver
    Controlled via sockets
    Adapted from QCoDeS Lakeshore 336 driver
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_336.A.temperature() and Model_336.channels[0].temperature()
        # refer to the same parameter.
        self.visa_handle.baud_rate = 57600
        self.visa_handle.stop_bits = visa.constants.StopBits.one
        self.visa_handle.parity = visa.constants.Parity.odd
        self.visa_handle.data_bits = 7
        channels = ChannelList(self, "TempSensors", SensorChannel, snapshotable=False)
        for chan_name in ('A'):
            channel = SensorChannel(self, 'Chan{}'.format(chan_name), chan_name)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)
        ###############
        self.add_parameter(name='set_temperature',
                   get_cmd='SETP?',
                   get_parser=float,
                   set_cmd='SETP 1,{}',
                   label='Set Temerature',
                   vals=Numbers(4, 300),
                   unit='K')
        self.add_parameter(name='heater_range',
                   get_cmd='RANGE?',
                   get_parser=int,
                   set_cmd='RANGE 1,{}',
                   label='Heater range',
                   vals=Enum(0, 1, 2, 3),
                   unit='')
        self.add_parameter(name='ramp_rate',
                   get_cmd='RAMP? 1',
                   get_parser=str,
                   set_cmd='RAMP 1,1,{}',
                   label='Heater range',
                   vals=Numbers(min_value=0),
                   unit='K/min')
        self.add_parameter(name='analog_output',
                   get_cmd='ANALOG?',
                   get_parser=str,
                   label='Analog output')
        ##############
        self.connect_message()