#: Various Python utilities
#from typing import Dict, List, Sequence, Any, Union, Tuple
import numpy as np
#import json
import time
#from nidaqmx.stream_readers import AnalogSingleChannelReader
#from nidaqmx._task_modules.in_stream import InStream as ins
#ins=InStream

#import time
#from IPython.display import clear_output
import matplotlib.pyplot as plt
#from qcodes.data.io import DiskIO
import pathlib

import sys
sys.path.append(r'C:\Users\ACSAFM\Dropbox (Moler group)\TeamFrieda\Programs\RedPitaya')
from pyrpl import Pyrpl

from scipy import io
#import nidaqmx

#from PyDAQmx import *

#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
#from plots import ScanPlot, TDCPlot
#from microscope.microscope import Microscope
#import utils

#: Pint for manipulating physical units
#from pint import UnitRegistry
#ureg = UnitRegistry()
#: Tell UnitRegistry instance what a Phi0 is, and that Ohm = ohm
#with open('squid_units.txt', 'w') as f:
#    f.write('Phi0 = 2.067833831e-15 * Wb\n')
#    f.write('Ohm = ohm\n')
#ureg.load_definitions('./squid_units.txt')

import logging
log = logging.getLogger(__name__)

class noise:

    def __init__(self) -> None:
         super().__init__( )
    #@staticmethod
    def noise_vs_IPHI():
        # sampling rate in Hz
        samplerate=1000000
        # sampling duration in seconds
        sampleduration=1
        # fmax in Hz
        fmax=1e4
        navg=1
        # minimum voltage across SQUID current leads
        v0_min=0.05
        # maximum voltage across SQUID current leads
        v0_max=0.3
        # minimum voltage across SQUID field coil leads
        v1_min=-10
        # maximum voltage across SQUID field coil leads
        v1_max=10
        # number of SQUID voltage points
        nv0s=40
        # number of field coil voltage points
        nv1s=40
        
        # volts per phi0
        vphi=3.4


        nsamples=int(samplerate*sampleduration)
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('f(Hz)')
        ax.set_ylabel('Phi_0/sqrt(Hz)')  
        v0_list=np.linspace(v0_min,v0_max,nv0s)
        v1_list=np.linspace(v1_min,v1_max,nv1s)
        p=Pyrpl('squid_lockbox')
        r=p.rp
        Iave_mat=np.zeros((nv0s,nv1s))
        Mave_mat=np.zeros((nv0s,nv1s))
        with nidaqmx.Task() as read_task:
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                samples_to_read=nsamples)
            
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao0')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                nv0=-1
                for v0 in v0_list:
                    nv0=nv0+1
                    nv1=-1
                    for v1 in v1_list:
                        nv1=nv1+1
                        print('V0='+str(v0)+'V V1='+str(v1)+'V\r')
                        write_task.start()
                        vout=[v0,
                              v1]
                        write_task.write(vout,auto_start=True)
                        write_task.stop()
                        # turn off RedPitaya feedback
                        print('Turn off feedback\r')
                        r.pid0.p,r.pid0.i = 0,0
                        # turn on modulation
                        print('Turn on modulation\r')
                        r.asg0.output_direct='out1'
                        r.asg0.setup(waveform='ramp', frequency=50, amplitude=0.8, offset=0, trigger_source='immediately')
                        r.asg0.output_direct='out2'
                        print('Measure average current\r')
                        # measure average current
                        Iave=0
                        for nt in range(1,3000):
                            Iave+=r.sampler.in1
                        Iave=Iave/3000
                        Iave_mat[nv0,nv1]=Iave
                        # set setpoint to average current
                        print('Setpoint to'+str(Iave)+'V\r')
                        r.pid0.setpoint=Iave
                        # turn off ac modulation
                        print('Turn off ac modulation\r')
                        r.asg0.output_direct="off"
                        # turn on feedback
                        print('Turn on feedback\r')
                        r.pid0.p=0
                        r.pid0.i=800
                        r.pid0.inputfilter=(0,0,0,0)
                        # reset lockpoint
                        r.pid0.ival=0
                        # take time-trace
                        print('Take time trace\r')
                        v_fft_avg = np.zeros((nsamples//2,))
                        for i in range(navg):
                            noise_data=daq_ai.voltage()[0].T
                            Mave_mat[nv0,nv1]=Mave_mat[nv0,nv1]+np.mean(noise_data)
                            noise_datar=np.multiply(noise_data,1/vphi)
                            noise_dataf=np.fft.fft(noise_datar)/(nsamples/np.sqrt(2*sampleduration))
                            v_fft_abs=np.abs(noise_dataf[:nsamples//2])
                            frequency_vec=np.fft.fftfreq(nsamples,d=1/samplerate)[:nsamples//2]
                            v_fft_avg+=v_fft_abs
                 
                        v_fft_avg=v_fft_avg/navg  
                        Mave_mat[nv0,nv1]=Mave_mat[nv0,nv1]/navg  
                        ax.loglog(frequency_vec[frequency_vec<fmax],v_fft_avg[frequency_vec<fmax]) 
                        #ax.plot(noise_datar[1:ntimes]) 
                        #plt.pause(0.1)
                        fig.canvas.draw()

                        fname='I'+str(nv0)+'Phi'+str(nv1)+'.dat'
                        print('Save data to '+fname+'\n\r')
                        f1=open(fname,'w')
                        for nfreq in range(1,nsamples//2):
                            if(frequency_vec[nfreq]<fmax):
                                f1.write(str(frequency_vec[nfreq])+','+str(v_fft_avg[nfreq])+'\n')
                        f1.close()  
                        #time.sleep(0.1) 
        daq_ai.close()
        np.savetxt('Iave.dat',Iave_mat)
        np.savetxt('Mave.dat',Mave_mat)
    def I_noise_vs_VPHI():
        # this measures the open loop current noise as a function of V and Phi
        # for these measurements the array amplifier should be locked using the RedPitaya
        # and the current output should go to ADC4 instead of the flux signal
        # sampling rate in Hz
        samplerate=1000000
        # sampling duration in seconds
        sampleduration=1
        # fmax in Hz
        fmax=1e4
        navg=1
        # minimum voltage across SQUID current leads
        v0_min=-0.7
        # maximum voltage across SQUID current leads
        v0_max=0.7
        # minimum voltage across SQUID field coil leads
        v1_min=-0.5
        # maximum voltage across SQUID field coil leads
        v1_max=0.5
        # number of SQUID voltage points
        nv0s=50
        # number of field coil voltage points
        nv1s=50
        
        nsamples=int(samplerate*sampleduration)
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('f(Hz)')
        ax.set_ylabel('Vi/sqrt(Hz)')  
        v0_list=np.linspace(v0_min,v0_max,nv0s)
        v1_list=np.linspace(v1_min,v1_max,nv1s)
        p=Pyrpl('squid_lockbox')
        r=p.rp
        Iave_mat=np.zeros((nv0s,nv1s))
        
        with nidaqmx.Task() as read_task:
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                samples_to_read=nsamples)
            
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao0')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                #nv0=29
                for nv0 in range(30,50):
                    v0=v0_list[nv0]
                    #nv0=nv0+1
                    #nv1=29
                    # reset lockpoint
                    r.pid0.ival=0
                    for nv1 in range(0,30):
                        v1=v1_list[nv1]
                        #nv1=nv1+1
                        print('V0='+str(v0)+'V V1='+str(v1)+'V\r')
                        write_task.start()
                        vout=[v0,
                              v1]
                        write_task.write(vout,auto_start=True)
                        write_task.stop()
                        
                        # take time-trace
                        print('Take time trace\r')
                        v_fft_avg = np.zeros((nsamples//2,))
                        for i in range(navg):
                            noise_data=daq_ai.voltage()[0].T
                            Iave_mat[nv0,nv1]=Iave_mat[nv0,nv1]+np.mean(noise_data)
                            noise_dataf=np.fft.fft(noise_data)/(nsamples/np.sqrt(2*sampleduration))
                            v_fft_abs=np.abs(noise_dataf[:nsamples//2])
                            frequency_vec=np.fft.fftfreq(nsamples,d=1/samplerate)[:nsamples//2]
                            v_fft_avg+=v_fft_abs
                 
                        v_fft_avg=v_fft_avg/navg  
                        Iave_mat[nv0,nv1]=Iave_mat[nv0,nv1]/navg
                        ax.loglog(frequency_vec[frequency_vec<fmax],v_fft_avg[frequency_vec<fmax]) 
                        #ax.plot(noise_datar[1:ntimes]) 
                        #plt.pause(0.1)
                        fig.canvas.draw()

                        fname='V'+str(nv0)+'Phi'+str(nv1)+'.dat'
                        print('Save data to '+fname+'\n\r')
                        f1=open(fname,'w')
                        for nfreq in range(1,nsamples//2):
                            if(frequency_vec[nfreq]<fmax):
                                f1.write(str(frequency_vec[nfreq])+','+str(v_fft_avg[nfreq])+'\n')
                        f1.close()  
                        #time.sleep(0.1) 
        daq_ai.close()
        np.savetxt('Iave.dat',Iave_mat)
           

    
    def IV_vs_PHI():
         # sampling rate in Hz
        samplerate=1000000
        # minimum voltage across SQUID current leads
        v0_min=-0.7
        # maximum voltage across SQUID current leads
        v0_max=0.7
        # minimum voltage across modulation coil leads
        v1_min=-0.5
        # maximum voltage across modulation coil leads
        v1_max=0.5
        # number of SQUID voltage points
        nv0s=50
        # number of modulation coil voltage points
        nv1s=50
        # number of averages of current signal
        nsamples=10000
        
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Vv')
        ax.set_ylabel('Vi')  
        v0_list=np.linspace(v0_min,v0_max,nv0s)
        v1_list=np.linspace(v1_min,v1_max,nv1s)
        p=Pyrpl('squid_lockbox')
        r=p.rp
        Iave_mat=np.zeros((nv0s,nv1s))
        # turn off RedPitaya feedback
        print('Turn on feedback\r')
        r.pid0.p=0
        r.pid0.i=800
        r.pid0.inputfilter=(0,0,0,0)
        # reset lockpoint
        r.pid0.ival=0

        with nidaqmx.Task() as read_task:
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                samples_to_read=nsamples)
        
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao0')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                nv1=-1
                for v1 in v1_list:
                    nv1=nv1+1
                    nv0=-1
                    print('V1='+str(v1)+'V\r')
                    # reset lockpoint
                    r.pid0.ival=0.25
                    for v0 in v0_list:
                        nv0=nv0+1
                        
                        write_task.start()
                        vout=[v0,
                              v1]
                        write_task.write(vout,auto_start=True)
                        write_task.stop()
                        
                        time.sleep(0.1)
                        #print('Measure average current\r')
                        # measure average current
                        I_data=daq_ai.voltage()[0].T
                        
                        Iave_mat[nv0,nv1]=np.mean(I_data)
                    ax.plot(v0_list,Iave_mat[:,nv1])    
            

                     
        daq_ai.close()
        np.savetxt('Iave.dat',Iave_mat)
        




            