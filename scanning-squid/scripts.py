from instruments.daq import DAQAnalogInputs
import qcodes as qc
from qcodes.data.io import DiskIO
import pathlib
import numpy as np
from scipy import io
import nidaqmx
import matplotlib.pyplot as plt

def fft_noise(dev_name, channel, prefactor, samplerate, sampleduration, navg, fmax):
    loc_provider = qc.FormatLocation(fmt='./data/{date}/#{counter}_{name}_{time}')
    loc = loc_provider(DiskIO('.'), record={'name': 'fft_noise'})
    pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
    prefactor_str = {}
    prefactor.ito('Phi0/V')
    prefactor_str.update({list(channel.keys())[0]: '{} {}'.format(prefactor.magnitude, prefactor.units)})
    mdict = {
        'metadata': {
            'channel': channel,
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
        daq_ai = DAQAnalogInputs('daq_ai', 'Dev1', samplerate, channel, ai_task, samples_to_read=nsamples)
        for i in range(navg):
            data_v = daq_ai.voltage()[0].T
            Fs = nsamples / sampleduration
            v_fft = np.fft.fft(data_v) / (nsamples / np.sqrt(2 * sampleduration))
            v_fft_abs = np.abs(v_fft[:nsamples//2])
            freqs = np.fft.fftfreq(nsamples, d=1/Fs)[:nsamples//2]
            v_fft_avg += v_fft_abs
        daq_ai.close()
        v_fft_avg = v_fft_avg / navg
        phi_fft_avg = prefactor.magnitude * v_fft_avg
        mdict.update({
            'v_fft_avg': v_fft_avg[freqs < fmax],
            'phi_fft_avg': phi_fft_avg[freqs < fmax],
            'freqs': freqs[freqs < fmax]})
        fig, ax = plt.subplots(1,2, figsize=(8,4), tight_layout=True)
        ax[0].loglog(freqs, v_fft_avg, lw=1)
        ax[0].set_ylabel(r'V/$\sqrt{Hz}$')
        ax[1].loglog(freqs, phi_fft_avg, lw=1)
        ax[1].set_ylabel(r'$\Phi_0/\sqrt{Hz}$')
        fig.suptitle(loc, x=0.5, y=1)
        for i in [0,1]:
            ax[i].set_xlabel('Frequency [Hz]')
            ax[i].grid()
        plt.savefig(loc + '/fft_noise.png')
        io.savemat(loc + '/fft_noise.mat', mdict)
        return mdict