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

from instruments.daq import DAQAnalogInputs
import qcodes as qc
from qcodes.data.io import DiskIO
import pathlib
import numpy as np
from scipy import io
import nidaqmx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Any, Union, Tuple

def fft_noise(dev_name: str, channel: Dict[str, int], unit: str,
              prefactor: Any, samplerate: int, sampleduration: Union[float,int],
              navg: int, fmax: Union[float,int]):
    """Noise measurement of a single channel.
    
    Args:
        dev_name: DAQ device name (e.g. 'Dev1').
        channel: Dict of {channel_name: analog_input} (e.g. {'MAG': 0}).
        unit: Physical unit of the channel (e.g. 'Phi0').
        prefactor: Pint Quantity with dimenions of unit/V, from microscope.get_prefactors().
        samplerate: DAQ sampling rate in Hz.
        sampleduration: Sampling time in seconds.
        navg: Number of times to average the spectrum.
        fmax: Maximum frequency up to which the spectrum will be saved.
        
    Returns:
        Dict: mdict
    """
    loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
    loc = loc_provider(DiskIO('.'), record={'name': 'fft_noise'})
    pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
    prefactor_str = {}
    prefactor.ito('{}/V'.format(unit))
    prefactor_str.update({list(channel.keys())[0]: '{} {}'.format(prefactor.magnitude, prefactor.units)})
    mdict = {
        'metadata': {
            'channel': channel,
            'unit': unit,
            'prefactor': prefactor_str,
            'samplerate': samplerate,
            'sampleduration': sampleduration,
            'navg': navg,
            'fmax': fmax ,
            'location': loc
        }
    }
    nsamples = int(samplerate * sampleduration)
    v_fft_avg = np.zeros((nsamples // 2,))
    with nidaqmx.Task('fft_noise_ai_task') as ai_task:
        for inst in DAQAnalogInputs.instances():
            inst.close()
        daq_ai = DAQAnalogInputs('daq_ai', dev_name, samplerate, channel, ai_task,
            samples_to_read=nsamples, timeout=sampleduration+10)
        for i in range(navg):
            data_v = daq_ai.voltage()[0].T
            Fs = nsamples / sampleduration
            v_fft = np.fft.fft(data_v) / (nsamples / np.sqrt(2 * sampleduration))
            v_fft_abs = np.abs(v_fft[:nsamples//2])
            freqs = np.fft.fftfreq(nsamples, d=1/Fs)[:nsamples//2]
            v_fft_avg += v_fft_abs
        daq_ai.close()
        v_fft_avg = v_fft_avg / navg
        sig_fft_avg = prefactor.magnitude * v_fft_avg
        mdict.update({
            'v_fft_avg': v_fft_avg[freqs < fmax],
            'sig_fft_avg': sig_fft_avg[freqs < fmax],
            'freqs': freqs[freqs < fmax]})
        fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
        ax[0].loglog(freqs, v_fft_avg, lw=1)
        ax[0].set_ylabel('V/$\\sqrt{Hz}$')
        ax[1].loglog(freqs, sig_fft_avg, lw=1)
        ax[1].set_ylabel('{}/$\\sqrt{Hz}$'.format(unit))
        fig.suptitle(loc, x=0.5, y=1)
        for i in [0,1]:
            ax[i].set_xlabel('Frequency [Hz]')
            ax[i].grid()
        plt.savefig(loc + '/fft_noise.png')
        io.savemat(loc + '/fft_noise.mat', mdict)
        return mdict

def time_trace(dev_name: str, channels: Dict[str,int], units: Dict[str,str],
               prefactors: Dict[str, Any], samplerate: int, sampleduration: Union[float,int]):
    """Records a time trace of data from DAQ analog input channels, converts data to desired units.
    
    Args:
        dev_name: DAQ device name (e.g. 'Dev1').
        channel: Dict of {channel_name: analog_input} (e.g. {'MAG': 0, 'SUSCX': 1}).
        unit: Physical unit of the channel (e.g. {'MAG': 'Phi0', 'SUSCX': 'Phi0/A'}).
        prefactor: Dict of {channel_name: Pint Quantity} from microscope.get_prefactors().
        samplerate: DAQ sampling rate (for each channel) in Hz.
        sampleduration: Sampling time in seconds.
        
    Returns:
        Dict: mdict
    """
    loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
    loc = loc_provider(DiskIO('.'), record={'name': 'time_trace'})
    pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
    prefactor_strs = {}
    for ch in channels:
        unit = units[ch]
        prefactors[ch].ito('{}/V'.format(unit))
        prefactor_strs.update({ch: '{} {}'.format(prefactors[ch].magnitude, prefactors[ch].units)})
    nsamples = int(samplerate * sampleduration)
    time = np.linspace(0, sampleduration, nsamples)
    mdict = {
        'time': {'array': time, 'unit': 's'},
        'metadata': {
            #'channels': channels,
            #'units': units,
            #'prefactors': prefactor_strs,
            'samplerate': samplerate,
            'sampleduration': sampleduration,
            'location': loc
        }
    }
    nsamples = int(samplerate * sampleduration)
    time = np.linspace(0, sampleduration, nsamples)
    with nidaqmx.Task('time_trace_ai_task') as ai_task:
        for inst in DAQAnalogInputs.instances():
            inst.close()
        daq_ai = DAQAnalogInputs('daq_ai', dev_name, samplerate, channels, ai_task,
                                 samples_to_read=nsamples, timeout=sampleduration+10)
        data_v = daq_ai.voltage()
        daq_ai.close()
    for i, ch in enumerate(channels):
        mdict.update({ch: {'array': data_v[i] * prefactors[ch].magnitude, 'unit': units[ch], 'prefactor': prefactor_strs[ch]}})
    io.savemat(loc + '/time_trace.mat', mdict)
    return mdict
