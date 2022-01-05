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
from qcodes.utils.validators import Enum, Strings, Numbers, Ints, MultiType
import visa

class SensorChannel33x(InstrumentChannel):
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
                           unit='V')  # TODO: This will vary based on sensor type
        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel), get_parser=int,
                           val_mapping={'OK': 0, 'Invalid Reading': 1, 'Temp Underrange': 16, 'Temp Overrange': 32,
                           'Sensor Units Zero': 64, 'Sensor Units Overrange': 128}, label='Sensor_Status')
        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=Strings(),
                           label='Sensor_Name')

class Model_331(VisaInstrument):
    """
    Lakeshore Model 331 Temperature Controller Driver
    Controlled via sockets
    Adapted from QCoDeS Lakeshore 336 driver
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_331.A.temperature() and Model_331.channels[0].temperature()
        # refer to the same parameter.

        # Serial parameters if instrument is connected via RS-232:
        # self.visa_handle.baud_rate = 57600
        # self.visa_handle.stop_bits = visa.constants.StopBits.one
        # self.visa_handle.parity = visa.constants.Parity.odd
        # self.visa_handle.data_bits = 7
        channels = ChannelList(self, "TempSensors", SensorChannel33x, snapshotable=False)
        for chan_name in ('A'):
            channel = SensorChannel33x(self, 'Chan{}'.format(chan_name), chan_name)
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
                   vals=Numbers(3, 300),
                   unit='K')
        self.add_parameter(name='heater_range',
                   get_cmd='RANGE?',
                   get_parser=int,
                   set_cmd='RANGE {}',
                   label='Heater range',
                   vals=Enum(0, 1, 2, 3),
                   unit='')
        self.add_parameter(name='ramp_rate',
                   get_cmd='RAMP? 1',
                   get_parser=str,
                   set_cmd='RAMP 1,1,{}',
                   label='Heater range',
                   vals=Numbers(min_value=0, max_value=100),
                   unit='K/min')
        ##############
        self.connect_message()

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
        # i.e. Model_335.A.temperature() and Model_335.channels[0].temperature()
        # refer to the same parameter.

        # Serial parameters if instrument is connected via RS-232:
        self.visa_handle.baud_rate = 57600
        self.visa_handle.stop_bits = visa.constants.StopBits.one
        self.visa_handle.parity = visa.constants.Parity.odd
        self.visa_handle.data_bits = 7
        channels = ChannelList(self, "TempSensors", SensorChannel33x, snapshotable=False)
        for chan_name in ('A'):
            channel = SensorChannel33x(self, 'Chan{}'.format(chan_name), chan_name)
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
        ##############
        self.connect_message()

class SensorChannel372(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """
    valid_channels = ('A',) + tuple('ch{}'.format(i) for i in range(1,17))
    _CHANNEL_VAL = Enum(*valid_channels)

    def __init__(self, parent, name, channel, sensor_name):
        super().__init__(parent, name)

        # Validate the channel value
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel  # Channel on the temperature controller. Can be 1-16

        # Add the various channel parameters
        self.add_parameter('temperature', get_cmd='KRDG? {}'.format(self._channel[2:]),
                           get_parser=float,
                           label='Temerature',
                           unit='K')
        self.add_parameter('sensor_raw', get_cmd='SRDG? {}'.format(self._channel[2:]),
                           get_parser=float,
                           label='Raw_Reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type
        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel[2:]), get_parser=int,
                           val_mapping={'OK': 0, 'CS Overload': 1, 'VCM Overload': 2, 'VMIX Overload': 4,
                           'VDIF Overload': 8, 'Resisance Overrange': 16, 'Resistance Underrange': 32,
                           'Temp Overrange': 64, 'Temp Underrange': 128}, label='Sensor_Status')
        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel[2:]),
                           get_parser=self._sensor_name_parser, set_cmd='INNAME {},\"{{}}\"'.format(self._channel[2:]),
                           vals=Strings(), label='Sensor_Name')
        self.sensor_name(sensor_name)

    def _sensor_name_parser(self, msg):
        return str(msg).strip()

class Model_372(VisaInstrument):
    """
    Lakeshore Model 372 Temperature Controller Driver
    Controlled via sockets
    Adapted from QCoDeS Lakeshore 336 driver
    """

    def __init__(self, name, address, active_channels={'A': 'sample'}, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_335.A.temperature() and Model_335.channels[0].temperature()
        # refer to the same parameter.

        # Serial parameters if instrument is connected via RS-232:
        # self.visa_handle.baud_rate = 57600
        # self.visa_handle.stop_bits = visa.constants.StopBits.one
        # self.visa_handle.parity = visa.constants.Parity.odd
        # self.visa_handle.data_bits = 7
        channels = ChannelList(self, "TempSensors", SensorChannel372, snapshotable=False)
        for chan_name, sensor_name in active_channels.items():
            channel = SensorChannel372(self, 'Chan{}'.format(chan_name), chan_name, sensor_name)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)
        ###############
        self.add_parameter(name='set_temperature',
                    get_cmd='SETP?',
                    get_parser=float,
                    set_cmd='SETP 0,{}',
                    label='Set Temerature',
                    vals=Numbers(0.05, 40),
                    unit='K')
        self.add_parameter(name='heater_range',
                    get_cmd='RANGE? 0',
                    get_parser=int,
                    set_cmd='RANGE 0, {}',
                    label='Sample Heater range',
                    vals=Enum(0, 1, 2, 3,4,5,6,7,8),
                    unit='')
        self.add_parameter(name='ramp_rate',
                    get_cmd='RAMP? 0',
                    get_parser=str,
                    set_cmd='RAMP 0,1,{}',
                    label='Heater range',
                    vals=Numbers(min_value=0),
                    unit='K/min')
        self.add_parameter(
                    name='heater_output',
                    get_cmd='HTR?',
                    get_parser=float,
                    label='Heater Output',
                    unit='%')
        ##############
        self.connect_message()

    def configure_analog_output(self, input_name, low_value, high_value, maunal_value=0,
        output=2, bipolar=0, source=1, mode=1):
        msg = 'ANALOG {},{},{},{},{},{},{},{}'.format(
            output, bipolar, mode, input_name, source, high_value, low_value, maunal_value)
        self.write(msg)

    def pid(self, p, i, d):
        msg = ' PID 0,{},{},{}'.format(p, i, d)
        self.write(msg)

class SensorChannel34x(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """
    valid_channels = ('A', 'B', 'C', 'D')
    _CHANNEL_VAL = Enum(*valid_channels)

    def __init__(self, parent, name, channel, sensor_name):
        super().__init__(parent, name)

        # Validate the channel value
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel  # Channel on the temperature controller. Can be 1-16

        # Add the various channel parameters
        self.add_parameter('temperature', get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temerature',
                           unit='K')
        self.add_parameter('sensor_raw', get_cmd='SRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Raw_Reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type
        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel), get_parser=int,
                           val_mapping={'OK': 0, 'Invalid Reading': 1, 'Old Reading': 2, 'Temp Underrange': 16,
                           'Temp Overrange': 32, 'Units Zero': 64, 'Units Overrange': 128}, label='Sensor_Status')
        # self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
        #                    get_parser=self._sensor_name_parser, set_cmd='INNAME {},\"{{}}\"'.format(self._channel),
        #                    vals=Strings(), label='Sensor_Name')
        # self.sensor_name(sensor_name)

    def _sensor_name_parser(self, msg):
        return str(msg).strip()

class Model_340(VisaInstrument):
    """
    Lakeshore Model 340 Temperature Controller Driver
    Controlled via sockets
    Adapted from QCoDeS Lakeshore 336 driver
    """

    def __init__(self, name, address, active_channels={'A': 'sample','B':'MXC'}, **kwargs):
        super().__init__(name, address, terminator="\n\r", **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_340.A.temperature() and Model_340.channels[0].temperature()
        # refer to the same parameter.

        # Serial parameters if instrument is connected via RS-232:
        # self.visa_handle.baud_rate = 57600
        # self.visa_handle.stop_bits = visa.constants.StopBits.one
        # self.visa_handle.parity = visa.constants.Parity.odd
        # self.visa_handle.data_bits = 7
        channels = ChannelList(self, "TempSensors", SensorChannel34x, snapshotable=False)
        for chan_name, sensor_name in active_channels.items():
            channel = SensorChannel34x(self, chan_name, chan_name, sensor_name)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)
        ###############
        self.add_parameter(
                name='set_temperature',
                get_cmd='SETP? 1',
                get_parser=float,
                set_cmd='SETP 1,{}',
                label='Set Temerature',
                vals=Numbers(0.1, 300),
                unit='K',
                snapshot_get=False
            )
        self.add_parameter(
                name='heater_range',
                get_cmd='RANGE?',
                get_parser=int,
                set_cmd='RANGE {}',
                label='Heater range',
                vals=Enum(0, 1, 2, 3, 4, 5),
                unit=''
            )
        self.add_parameter(
                name='ramp_rate',
                get_cmd='RAMP? 1',
                get_parser=str,
                set_cmd='RAMP 1,1,{}',
                label='Heater range',
                vals=Numbers(min_value=0),
                unit='K/min'
            )
        self.add_parameter(
                name='analog_out_config',
                get_cmd='ANALOG? 1',
                get_parser=str,
                label='Analog output configuration.',
                unit=''
            )
        self.add_parameter(
                name='pid_loop1',
                get_cmd='PID? 1',
                get_parser=str,
                label='PID loop1',
                unit=''
            )
        self.add_parameter(
                name='heater_output',
                get_cmd='HTR?',
                get_parser=float,
                label='Heater Output',
                unit='%'
          )
        #############
        self.connect_message()

    def configure_analog_output(self, input_name, low_value, high_value, maunal_value=0,
        output=1, bipolar=0, source=1, mode=1):
        msg = 'ANALOG {},{},{},{},{},{},{},{}'.format(
            output, bipolar, mode, input_name, source, high_value, low_value, maunal_value)
        self.write(msg)

    def pid(self, loop_name, p, i, d):
        msg = ' PID {},{},{},{}'.format(loop_name, p, i, d)
        self.write(msg)