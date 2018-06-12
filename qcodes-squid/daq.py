from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ArrayParameter
from nidaqmx.constants import AcquisitionType, TaskMode
from typing import Dict, Optional, Sequence, Any, Union
import numpy as np

class DAQAnalogInputVoltages(ArrayParameter):
    """Acquires data from one or several DAQ analog inputs.
    """
    def __init__(self, name: str, task: Any, samples_to_read: int,
                 shape: Sequence[int], **kwargs) -> None:
        """
        Args:
            name: Name of parameter (usually 'voltage').
            task: nidaqmx.Task with appropriate analog inputs channels.
            samples_to_read: Number of samples to read. Will be averaged based on shape.
            shape: Desired shape of averaged array, i.e. (nchannels, desired_points).
            **kwargs: Keyword arguments to be passed to ArrayParameter constructor.
        """
        super().__init__(name, shape, **kwargs)
        self.task = task
        self.nchannels, self.target_points = shape
        self.samples_to_read = samples_to_read
        
    def get_raw(self):
        """Averages data to get `self.target_points` points per channel.
        If self.target_points == self.samples_to_read, no averaging is done.
        """
        data_raw = np.array(self.task.read(number_of_samples_per_channel=self.samples_to_read))
        return np.mean(np.reshape(data_raw, (self.nchannels, self.target_points, -1)), 2)
    
class DAQAnalogInputs(Instrument):
    """Instrument to acquire DAQ analog input data in a qcodes Loop or measurement.
    """
    def __init__(self, name: str, dev_name: str, rate: Union[int, float], channels: Dict[str, int],
                 task: Any, clock_src: Optional[str]=None, samples_to_read: Optional[int]=2,
                 target_points: Optional[int]=None, **kwargs) -> None:
        """
        Args:
            name: Name of instrument (usually 'daq_ai').
            dev_name: NI DAQ device name (e.g. 'Dev1').
            rate: Desired DAQ sampling rate in Hz.
            channels: Dict of analog input channel configuration.
            task: fresh nidaqmx.Task to be populated with ai_channels.
            clock_src: Sample clock source for analog inputs. Default: None
            samples_to_read: Number of samples to acquire from the DAQ
                per channel per measurement/loop iteration.
                Default: 2 (minimum number of samples DAQ will acquire in this timing mode).
            target_points: Number of points per channel we want in our final array.
                samples_to_read will be averaged down to target_points.
            **kwargs: Keyword arguments to be passed to Instrument constructor.
        """
        super().__init__(name, **kwargs)
        if target_points is None:
            if samples_to_read == 2: #: minimum number of samples DAQ will read in this timing mode
                target_points = 1
            else:
                target_points = samples_to_read
        self.rate = rate
        nchannels = len(list(channels.keys()))
        self.samples_to_read = samples_to_read
        self.task = task
        self.metadata.update({
            'dev_name': dev_name,
            'rate': '{} Hz'.format(rate),
            'channels': channels})
        for ch, idx in channels.items():
            channel = '{}/ai{}'.format(dev_name, idx)
            self.task.ai_channels.add_ai_voltage_chan(channel, ch)
        if clock_src is None:
            #: Use default sample clock timing: ai/SampleClockTimebase
            self.task.timing.cfg_samp_clk_timing(
                rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=samples_to_read)
        else:
            #: Clock the inputs on some other clock signal, e.g. 'ao/SampleClock'
            self.task.timing.cfg_samp_clk_timing(
                rate,
                source=clock_src,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=samples_to_read)
        #: We need a parameter in order to acquire voltage in a qcodes Loop or Measurement
        self.add_parameter(name='voltage',
                           parameter_class=DAQAnalogInputVoltages,
                           task=self.task,
                           samples_to_read=samples_to_read,
                           shape=(nchannels, target_points),
                           label='Voltage',
                           unit='V'
                          ) 
        
    def clear_instances(self):
        """Clear instances of DAQAnalogInputs Instruments.
        """
        for instance in self.instances():
            self.remove_instance(instance)
