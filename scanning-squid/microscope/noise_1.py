#: Various Python utilities
#from typing import Dict, List, Sequence, Any, Union, Tuple
import numpy as np
import json
import time
from nidaqmx.stream_readers import AnalogSingleChannelReader
#import time
#from IPython.display import clear_output
import matplotlib.pyplot as plt


#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot
from microscope.microscope import Microscope
import utils

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

class noise:

    def __init__(self) -> None:
         super().__init__( )
    #@staticmethod
    def noise_vs_IPHI():
        # sampling rate in Hz
        fSampling=10000
        nchannels = 1
        # volts per phi0
        vphi=3.4
        ntimes=10000
        dt=nchannels/fSampling
        T=dt*ntimes/2
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('f(Hz)')
        ax.set_ylabel('Phi_0/sqrt(Hz)')
        frequency_vec=np.linspace(-0.5,0.5,ntimes)/dt  
        v1_list=np.linspace(-0.01,0.01,3)
        v2_list=np.linspace(-0.01,0.01,3)
        noise_data = np.zeros(ntimes, dtype=np.float64)
        #noise_data = np.zeros(ntimes)
        with nidaqmx.Task() as read_task:
            read_task.ai_channels.add_ai_voltage_chan("Dev2/ai4")
            read_task.timing.cfg_samp_clk_timing(fSampling)
            reader = AnalogSingleChannelReader(read_task.in_stream)
            #read_task.start()
            #task1.wait_until_done(10.0)
            #task1.DAQmxCfgInputBuffer(task1,10000)
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan("Dev2/ao0")
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                for v1 in v1_list:
                    for v2 in v2_list:
                        #write_task.start
                        #write_task.write([v1,v2],auto_start=True)
                        #write_task.stop
                        #noise_data = np.zeros(ntimes, dtype=np.float64)
                        #read_task.start()
                        reader.read_many_sample(noise_data, number_of_samples_per_channel=ntimes,timeout=2)
                        read_task.stop()
                        #noise_data=task1.read(number_of_samples_per_channel=ntimes)
                        noise_datar=np.multiply(noise_data,1/vphi)
                        noise_dataf=np.multiply(np.fft.fft(noise_datar),np.sqrt(T)/ntimes)
                        ax.plot(frequency_vec[int(ntimes/2)+1:ntimes],np.real(noise_dataf)[int(ntimes/2)+1:ntimes]) 
                        plt.pause(0.1)
                        #fig.canvas.draw_idle()
                        fig.canvas.draw
                        fname='v1'+str(v1)+'v2'+str(v2)+'.dat'
                        f1=open(fname,'w')
                        for nfreq in range(int(ntimes/2)+1,ntimes):
                            f1.write(str(frequency_vec[nfreq])+','+str(np.real(noise_dataf[nfreq]))+'\n')
                        f1.close()  
    def IV_vs_Phi():
        nvs=100
        nphis=100
        v_list=np.linspace(-1,1,nvs)
        phi_list=np.linspace(-1,1,nphis)
        current_data = np.zeros(nvs, dtype=np.float64)
        #noise_data = np.zeros(ntimes)
        with nidaqmx.Task() as read_task:
            read_task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
            read_task.timing.cfg_samp_clk_timing(fSampling)
            reader = AnalogSingleChannelReader(read_task.in_stream)
            #read_task.start()
            #task1.wait_until_done(10.0)
            #task1.DAQmxCfgInputBuffer(task1,10000)
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan("Dev2/ao0")
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                for v1 in v1_list:
                    for v2 in v2_list:
                        write_task.write([v1,v2],auto_start=True)
                        #noise_data = np.zeros(ntimes, dtype=np.float64)
                        read_task.start()
                        reader.read_many_sample(noise_data, number_of_samples_per_channel=ntimes,timeout=2)
                        read_task.stop()
                        #noise_data=task1.read(number_of_samples_per_channel=ntimes)
                        noise_datar=np.multiply(noise_data,1/vphi)
                        noise_dataf=np.multiply(np.fft.fft(noise_datar),np.sqrt(T)/ntimes)
                        ax.plot(frequency_vec[int(ntimes/2)+1:ntimes],np.real(noise_dataf)[int(ntimes/2)+1:ntimes]) 
                        plt.pause(0.1)
                        fig.canvas.draw_idle()
                        fname='v1'+str(v1)+'v2'+str(v2)+'.dat'
                        f1=open(fname,'w')
                        for nfreq in range(int(ntimes/2)+1,ntimes):
                            f1.write(str(frequency_vec[nfreq])+','+str(np.real(noise_dataf[nfreq]))+'\n')
                        f1.close()  


            