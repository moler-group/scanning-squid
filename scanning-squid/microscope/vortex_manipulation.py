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

class vortex:

    def __init__(self) -> None:
         super().__init__( )
    #@staticmethod
    def mag_vs_Ifc():
        # this plots the mag signal vs the current through the field coil
         # sampling rate in Hz
        samplerate=1000000
        # minimum voltage across field coil leads
        vfc_min=-10
        # maximum voltage across field coil leads
        vfc_max=10
        # number of field coil voltage points
        nvfcs=100
        # number of averages of mag signal
        nsamples=1000
        mag_list=np.zeros(4*nvfcs)
        
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Vfc(V)')
        ax.set_ylabel('Vmag(V)') 
        vfc_list=np.zeros(4*nvfcs) 
        print(len(vfc_list))
        for npv in range (0,nvfcs):
            vfc_list[npv]=vfc_max*npv/nvfcs
        for npv in range (nvfcs,3*nvfcs):
            vfc_list[npv] = vfc_max+(vfc_min-vfc_max)*(npv-nvfcs)/(2*nvfcs)
        for npv in range (3*nvfcs,4*nvfcs):        
            vfc_list[npv] = vfc_min-(vfc_min)*(npv-3*nvfcs)/(nvfcs)
                 
        
        # this is a test. This is only a test. If this were a real emergency you are toast.
        with nidaqmx.Task() as read_task:
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                samples_to_read=nsamples)
        
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao3')
                nvfc=-1
                for vfc in vfc_list:
                    write_task.start()
                    write_task.write(vfc,auto_start=True)
                    write_task.stop()
                    time.sleep(0.01)
                    nvfc=nvfc+1
                    #print('Measure average current\r')
                    # measure average current
                    mag_data=daq_ai.voltage()[0].T
                    mag_list[nvfc]=np.mean(mag_data)
                #ax.plot(vfc_list,mag_list)  
                #nvfc=-1
                #for vfc in vfc_list:
                #write_task.start()
                #write_task.write(vfc_list,auto_start=True)
                #write_task.stop()
                #    time.sleep(0.01)
                #    nvfc=nvfc+1
                    #print('Measure average current\r')
                    # measure average current
                #mag_list=daq_ai.voltage()[0].T
                #mag_list=np.mean(mag_data)
                ax.plot(vfc_list,mag_list)      
#            
#
#                     
        daq_ai.close()
        np.savetxt('mag11p7iK.dat',(vfc_list,mag_list))
        

    def scan_mag_vs_Ifc():
             # this plots the mag signal vs the current through the field coil at a number of x,y positions
              # sampling rate in Hz
        samplerate=1000000
             # minimum voltage across field coil leads
        vfc_min=-5
             # maximum voltage across field coil leads
        vfc_max=5
             # number of field coil voltage points
        nvfcs=50
             # number of averages of mag signal
        nsamples=1000
             # minimum x-piezo voltage in volts
        xv_min=0
             # maximum x-pizeo voltage in volts
        xv_max=1.5
             # minimum y-piezo voltage in volts
        yv_min=-0.3
             # maximum y-piezo voltage in volts
        yv_max=1.2
             # number of x pixels
        nxps=30
             # number of y pixels
        nyps=30
             # absolute maximum allowable piezo voltage in volts 
        vmax=4
             # x-slope 
        xslope=0.084
             # y-slope
        yslope=-0.025
             # z-touchdown voltage in volts
        ztouch=5.465
             # scan height in volts
        height=-0.2
             # maximum allowed z-piezo voltage in volts
        zmax=7.5
        xv_list=np.linspace(xv_min,xv_max,nxps)
        yv_list=np.linspace(yv_min,yv_max,nyps)
        X=np.zeros((nyps,nxps))
        Y=np.zeros((nyps,nxps))
        Z=np.zeros((nyps,nxps))
        mag_list=np.zeros(4*nvfcs)
        for nxv in range(0,nxps):
            for nyv in range(0,nyps):
                X[nyv,nxv]=xv_list[nxv]
                Y[nyv,nxv]=yv_list[nyv]
                Z[nyv,nxv]=ztouch+height+xslope*xv_list[nxv]+yslope*yv_list[nyv]
        if np.max(np.abs(Z)) > zmax:
            printf('Z voltage exceeds maximum allowed 7.5V')
            return()    

        if np.max(np.abs(xv_list)) > vmax or np.max(np.abs(yv_list)) > vmax:
            printf('X or Y voltage exceeds maximum allowed 4V')
            return()

            
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Vfc(V)')
        ax.set_ylabel('Vmag(V)')  
             # make the field coil sweep from 0 to max, then to min, then back to 0
        #vfc_list1=np.linspace(0,vfc_max,nvfcs)
        #vfc_list2=np.linspace(vfc_max,vfc_min,2*nvfcs)
        #vfc_list3=np.linspace(vfc_min,0,nvfcs)
        #vfc_list=vfc_list1+vfc_list2+vfc_list3
        vfc_list=np.zeros(4*nvfcs) 
        for npv in range (0,nvfcs):
            vfc_list[npv]=vfc_max*npv/nvfcs
        for npv in range (nvfcs,3*nvfcs):
            vfc_list[npv] = vfc_max+(vfc_min-vfc_max)*(npv-nvfcs)/(2*nvfcs)
        for npv in range (3*nvfcs,4*nvfcs):        
            vfc_list[npv] = vfc_min-(vfc_min)*(npv-3*nvfcs)/(nvfcs)    

        
        with nidaqmx.Task() as read_task:
            for inst in DAQAnalogInputs.instances():
                inst.close()
            daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                samples_to_read=nsamples)    
            with nidaqmx.Task() as write_task:   
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao0')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao1')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao2')
                write_task.ao_channels.add_ao_voltage_chan('Dev2/ao3')
                for nyv in range(0,nxps):
                    for nxv in range(0,nyps):
                        nvfc=-1
                        for vfc in vfc_list:
                            nvfc=nvfc+1
                            Vout_list=[X[nyv,nxv],
                                    Y[nyv,nxv],
                                    Z[nyv,nxv],
                                    vfc]
                            write_task.start()
                            write_task.write(Vout_list,auto_start=True)
                            write_task.stop()
                            time.sleep(0.01)
                            mag_data=daq_ai.voltage()[0].T
                            mag_list[nvfc]=np.mean(mag_data)
                        ax.plot(vfc_list,mag_list)    
                        fname='X'+str(nxv)+'Y'+str(nyv)+'.dat'
                         #f1=open(fname,'w')
                        np.savetxt(fname,(vfc_list,mag_list),fmt='%8.4e',newline='\n')
                         #f1.close()  
        daq_ai.close()
            



            