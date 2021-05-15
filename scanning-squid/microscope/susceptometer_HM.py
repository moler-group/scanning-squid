#: Various Python utilities
from typing import Dict, List, Sequence, Any, Union, Tuple
import numpy as np
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pdb
import scipy.io as io

#: Qcodes for running measurements and saving data
import qcodes as qc

#: NI DAQ library
import nidaqmx
from nidaqmx.constants import AcquisitionType

#: scanning-squid modules
from instruments.daq import DAQAnalogInputs
from plots import ScanPlot, TDCPlot
from microscope.microscope_HM import Microscope
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

class SusceptometerMicroscope(Microscope):
    """Scanning SQUID susceptometer microscope class.
    """
    def __init__(self, config_file: str, temp: str, ureg: Any=ureg, log_level: Any=logging.INFO,
                 log_name: str=None, **kwargs) -> None:
        """
        Args:
            config_file: Path to microscope configuration JSON file.
            temp: 'LT' or 'RT', depending on whether the microscope is cold or not.
                Sets the voltage limits for the scanner and Attocubes.
            ureg: pint UnitRegistry for managing physical units.
            log_level: e.g. logging.DEBUG or logging.INFO
            log_name: Log file will be saved as logs/{log_name}.log.
                Default is the name of the microscope configuration file.
            **kwargs: Keyword arguments to be passed to Station constructor.
        """
        super().__init__(config_file, temp, ureg, log_level, log_name, **kwargs)

    def get_prefactors(self, measurement: Dict[str, Any], update: bool=True) -> Dict[str, Any]:
        """For each channel, calculate prefactors to convert DAQ voltage into real units.

        Args:
            measurement: Dict of measurement parameters as defined
                in measurement configuration file.
            update: Whether to query instrument parameters or simply trust the
                latest values (should this even be an option)?

        Returns:
            Dict[str, pint.Quantity]: prefactors
                Dict of {channel_name: prefactor} where prefactor is a pint Quantity.
        """
        mod_width = self.Q_(self.SQUID.metadata['modulation_width'])
        prefactors = {}
        for ch in measurement['channels']:
            prefactor = 1
            if ch == 'MAG':
                prefactor /= mod_width
            elif ch == 'SUSCX':
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                amp = (self.SUSC_lockin.sigout_amplitude() *
                    self.SUSC_lockin.sigout_range() * self.ureg('V'))
                #: sqrt(2) because auxouts are Vpk, not Vrms
                prefactor *= np.sqrt(2) * (r_lead / amp) / (mod_width * self.SUSC_lockin.gain_X())
            elif ch == 'SUSCY':
                snap_susc = getattr(self, 'SUSC_lockin').snapshot(update=update)['parameters']
                r_lead = self.Q_(measurement['channels'][ch]['r_lead'])
                amp = (self.SUSC_lockin.sigout_amplitude() *
                    self.SUSC_lockin.sigout_range() * self.ureg('V'))
                #: sqrt(2) because auxouts are Vpk, not Vrms
                prefactor *=  np.sqrt(2) * (r_lead / amp) / (mod_width * self.SUSC_lockin.gain_Y())
            elif ch == 'CAP':
                gain_cap = self.CAP_lockin.gain_X()
                #: sqrt(2) because auxouts are Vpk, not Vrms
                prefactor *= np.sqrt(2) / (self.Q_(self.scanner.metadata['cantilever']['calibration']) * gain_cap)
            elif ch in ['x_cap', 'y_cap']:
                prefactor *= self.Q_(measurement['channels'][ch]['conversion'])
            prefactor /= measurement['channels'][ch]['gain']
            prefactors.update({ch: prefactor})
        return prefactors

    def scan_plane(self, scan_params: Dict[str, Any]) -> Any:
        """
        Scan the current plane while acquiring data in the channels defined in
        measurement configuration file (e.g. MAG, SUSCX, SUSCY, CAP).

        Args:
            scan_params: Dict of scan parameters as defined
                in measuremnt configuration file.

        Returns:
            Tuple[qcodes.DataSet, plots.ScanPlot]: data, plot
                qcodes DataSet containing acquired arrays and metdata,
                and ScanPlot instance populated with acquired data.
        """
        #if not self.atto.plane_is_current:
        #    raise RuntimeError('Plane is not current. Aborting scan.')
        old_pos = self.scanner.position()
        
        daq_config = self.config['instruments']['daq']
        #ao_channels = daq_config['channels']['analog_outputs']
        ao_channels = {}
        for ax in ['x', 'y', 'z']:
            ao_channels.update({ax: daq_config['channels']['analog_outputs'][ax]})
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = scan_params['channels']
        channels = {}
        for ch in meas_channels:
            channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())

        daq_name = daq_config['name']
        #: DAQ AI sampling rate is divided amongst all active AI channels
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels
        
        fast_ax = scan_params['fast_ax'].lower()
        slow_ax = 'x' if fast_ax == 'y' else 'y'
        
        line_duration = self.Q_(scan_params['range'][fast_ax]) / self.Q_(scan_params['scan_rate'])
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        pix_per_line = scan_params['scan_size'][fast_ax]
        
        plane = self.scanner.metadata['plane']
        height = self.Q_(scan_params['height']).to('V').magnitude
        scanner_constants = self.config['instruments']['scanner']['constants']
        scan_vectors = utils.make_scan_vectors(scan_params, scanner_constants, self.temp, self.ureg)
        scan_grids = utils.make_scan_grids(scan_vectors, slow_ax, fast_ax,
                                           pts_per_line, plane, height)
        utils.validate_scan_params(self.scanner.metadata, scan_params,
                                   scan_grids, self.temp, self.ureg, log)
        #pdb.set_trace()
        self.scanner.goto([scan_grids[axis][0][0] for axis in ['x', 'y', 'z']])
        # let the piezos relax before starting the scan
        time.sleep(30)
        #self.set_lockins(scan_params)
        #: get channel prefactors in pint Quantity form
        prefactors = self.get_prefactors(scan_params)
        #: get channel prefactors in string form so they can be saved in metadata
        prefactor_strs = {}
        for ch, prefac in prefactors.items():
            unit = scan_params['channels'][ch]['unit']
            pre = prefac.to('{}/V'.format(unit))
            prefactor_strs.update({ch: '{} {}'.format(pre.magnitude, pre.units)})
        ai_task = nidaqmx.Task('scan_plane_ai_task')
        self.remove_component('daq_ai')
        if hasattr(self, 'daq_ai'):
            self.daq_ai.clear_instances()
        self.daq_ai = DAQAnalogInputs('daq_ai', daq_name, daq_rate, channels, ai_task,
                                      samples_to_read=pts_per_line, target_points=pix_per_line,
                                      #: Very important to synchronize AOs and AIs
                                      clock_src='ao/SampleClock')
        self.add_component(self.daq_ai)
        slow_ax_position = getattr(self.scanner, 'position_{}'.format(slow_ax))
        slow_ax_start = scan_vectors[slow_ax][0]
        slow_ax_end = scan_vectors[slow_ax][-1]
        slow_ax_step = scan_vectors[slow_ax][1] - scan_vectors[slow_ax][0]
        #: There is probably a counter built in to qc.Loop, but I couldn't find it
        loop_counter = utils.Counter()
        scan_plot = ScanPlot(scan_params, scanner_constants, self.temp, self.ureg)
        loop = qc.Loop(slow_ax_position.sweep(start=slow_ax_start,
                                              stop=slow_ax_end,
                                              step=slow_ax_step), delay=0.1
        ).each(
            #: Create AO task and queue data to be written to AOs
            qc.Task(self.scanner.scan_line, scan_grids, ao_channels, daq_rate, loop_counter),
            #: Start AI task; acquisition won't start until AO task is started
            qc.Task(ai_task.start),
            #: Start AO task
            qc.Task(self.scanner.control_ao_task, 'start'),
            #: Acquire voltage from all active AI channels
            self.daq_ai.voltage,
            qc.Task(ai_task.wait_until_done),
            qc.Task(self.scanner.control_ao_task, 'wait_until_done'),
            qc.Task(ai_task.stop),
            #: Stop and close AO task so that AOs can be used for goto
            qc.Task(self.scanner.control_ao_task, 'stop'),
            qc.Task(self.scanner.control_ao_task, 'close'),
            qc.Task(self.scanner.goto_start_of_next_line, scan_grids, loop_counter),
            #: Update and save plot
            qc.Task(scan_plot.update, qc.loops.active_data_set, loop_counter),
            qc.Task(scan_plot.save),
            qc.Task(loop_counter.advance)
        ).then(
            qc.Task(ai_task.stop),
            qc.Task(ai_task.close),
            qc.Task(self.daq_ai.clear_instances),
            qc.Task(self.scanner.goto, old_pos)
        )
        # query temperature to be saved in snapshot
        self.temp_controller.A.temperature()    
        #: loop.metadata will be saved in DataSet
        loop.metadata.update(scan_params)
        loop.metadata.update({'prefactors': prefactor_strs})
        for idx, ch in enumerate(meas_channels):
            loop.metadata['channels'][ch].update({'idx': idx})
        data = loop.get_data_set(name=scan_params['fname'])
        #: Run the loop
        try:
            loop.run()
            log.info('Scan completed. DataSet saved to {}.'.format(data.location))
        #: If loop is aborted by user:
        except KeyboardInterrupt:
            log.warning('Scan interrupted by user. Going to [0, 0, 0] V.')
            self.abort_scan_loop = True
            #: Stop 'scan_plane_ai_task' so that we can read our current position
            ai_task.stop()
            ai_task.close()
            #: If there's an active AO task, close it so that we can use goto
            try:
                self.scanner.control_ao_task('stop')
                self.scanner.control_ao_task('close')
            except:
                pass
            self.scanner.goto([0, 0, 0])
            #self.CAP_lockin.amplitude(0.004)
            #self.SUSC_lockin.amplitude(0.004)
            log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
        utils.scan_to_mat_file(data, real_units=True)
        return data, scan_plot

    def multi_fc_scan(self, mfs_params: Dict[str, Any], scan_params: Dict[str, Any]):
        log.info('Starting multiple field-cooling scans.')
        self.abort_scan_loop = False
        self.keithley.mode('CURR')
        time.sleep(1)
        #self.keithley.sense('VOLT')
        self.keithley.rangei(100e-3)
        time.sleep(1)
        self.keithley.compliancev(10)
        time.sleep(1)
        self.keithley.curr(mfs_params['current'][0])
        time.sleep(1)
        self.keithley.output(1)
        for current in mfs_params['current']:
            if not self.abort_scan_loop:
                log.info('Setting current to {} A'.format(current))
                self.keithley.curr(current)
                self.keithley.volt()
                time.sleep(1)
                self.cycle_T(mfs_params['t_low'], mfs_params['t_high'])
                _ = self.scan_plane(scan_params)
                clear_output(wait=True)
    def multi_T_scan(self, mts_params: Dict[str, Any], scan_params: Dict[str, Any], td_params: Dict[str, Any]):
        log.info('Starting multiple temperature scans.')
        self.abort_scan_loop = False
        
        for temperature in mts_params['temperatures']:
            if not self.abort_scan_loop:
                log.info('Setting temperature to {} K'.format(temperature))
                self.temp_controller.ramp_rate(0)
                time.sleep(0.1)
                self.temp_controller.set_temperature(temperature)
                time.sleep(0.1)
                self.temp_controller.heater_range(mts_params['heater_range'])
                time.sleep(100)
                if mts_params['touchdown']==1:
                    _ = self.td_cap(td_params)
                _ = self.scan_plane(scan_params)
                clear_output(wait=True)   

    def multi_Vfc_scan(self, mts_params: Dict[str, Any], scan_params: Dict[str, Any]):
        log.info('Starting multiple field coil current scans.')
        self.abort_scan_loop = False
        
        for Vfc in mts_params['Vfcs']:
            if not self.abort_scan_loop:
                log.info('Setting susc lock-in amplitude to {} V'.format(Vfc))
                self.SUSC_lockin.sigout_amplitude(Vfc)
                time.sleep(0.1)
                _ = self.scan_plane(scan_params)
                clear_output(wait=True)

    def multi_fc_frequency_scan(self, mts_params: Dict[str, Any], scan_params: Dict[str, Any]):
        log.info('Starting multiple field coil frequency scans.')
        self.abort_scan_loop = False
        
        for frequency in mts_params['frequencies']:
            if not self.abort_scan_loop:
                log.info('Setting susc lock-in frequency to {} Hz'.format(frequency))
                self.SUSC_lockin.frequency(frequency)
                time.sleep(0.1)
                _ = self.scan_plane(scan_params)
                clear_output(wait=True)                           
    
    def multi_height_scan(self, mts_params: Dict[str, Any], scan_params: Dict[str, Any]):
        log.info('Starting multiple height scans.')
        self.abort_scan_loop = False
        for height in mts_params['heights']:
            if not self.abort_scan_loop:
                log.info('Setting height to {} '.format(height))
                scan_params['height']=height
                time.sleep(0.1)
                _ = self.scan_plane(scan_params)
                clear_output(wait=True)                                      
        
    def cycle_T(self, t_low: float, t_high: float):
        log.info('Starting temperature cycle. Warming to {}.'.format(t_high))
        try:
            self.temp_controller.ramp_rate(0)
            time.sleep(0.1)
            self.temp_controller.set_temperature(t_high)
            time.sleep(0.1)
            self.temp_controller.heater_range(2)
            for _ in range(90):
                time.sleep(1)
            self.temp_controller.ramp_rate(0.1)
            time.sleep(0.1)
            self.temp_controller.set_temperature(t_low)
            dt = t_high - t_low
            ramp_time = int(60 * dt / 0.1 + 10)
            for _ in range(ramp_time):
                time.sleep(1)
            self.temp_controller.heater_range(0)
            t = self.temp_controller.A.temperature()
            log.info('Temperature cycle complete. Current temperature: {} K.'.format(t))
        except KeyboardInterrupt:
            log.warning('Temperature cycle interrupted by user. Turning off heater.')
            self.temp_controller.heater_range(0)
    def plot_T_vs_time(self):
        time_vec=[]
        sample_T=[]
        elapsed_time=0
        self.fig, self.ax = plt.subplots(1,1)
        self.ax.set_xlabel('t(sec)')
        self.ax.set_ylabel('T(K)')
        
        while (True):
            sample_T.append(self.temp_controller.A.temperature())
            time_vec.append(elapsed_time)
            time.sleep(1)
            elapsed_time=elapsed_time+1
            self.ax.plot(time_vec,sample_T)
            self.fig.canvas.draw()
            
    #def plot_T_vs_time(self):
        #time_vec=[]
        #sample_T=[]
        #heater_V=[]
        #elapsed_time=0
        #self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        
        #while (True):
            #try:
                #sample_T.append(self.temp_controller.A.temperature())
                #heater_V.append(self.keithley.volt())
                #time_vec.append(elapsed_time)
                #time.sleep(1)
                #elapsed_time=elapsed_time+1
                #self.ax1.clear()
                #self.ax2.clear()
                #self.ax1.plot(time_vec,sample_T)
                #self.ax2.plot(time_vec,heater_V)
                #self.ax1.set_title('Temperature')
                #self.ax2.set_title('Heater')
                #self.ax1.set_xlabel('t(sec)')
                #self.ax1.set_ylabel('T(K)')
                #self.ax2.set_xlabel('t(sec)')
                #self.ax2.set_ylabel('Volt (V)')
                #self.fig.canvas.draw()
            #except KeyboardInterrupt:
                #print("Plotting interrupted by user.")
                #svdata = {'time_vec': time_vec,'sample_T': sample_T, 'heater_V': heater_V}
                #io.savemat('T_and_Heater_vs_Time.mat', svdata)
                #return svdata

        

    def IV_vs_temperature(self, mivts_params: Dict[str, Any]):
         # sampling rate in Hz
        samplerate=1000000
        
        nsamples=10000
        
        fig, ax = plt.subplots(1,1)
        ax.set_xlabel('Vv')
        ax.set_ylabel('Vi') 
        v0_min=mivts_params['v0_min']
        v0_max=mivts_params['v0_max']
        nv0s=mivts_params['nv0s']
        temperatures=mivts_params['temperatures']
        v0_list=np.linspace(v0_min,v0_max,nv0s)
        nTs=len(temperatures)
        #v1_list=np.linspace(v1_min,v1_max,nv1s)
        #p=Pyrpl('squid_lockbox')
        #r=p.rp
        Iave_mat=np.zeros((nv0s,nTs))
        # turn off RedPitaya feedback
        #print('Turn on feedback\r')
        #r.pid0.p=0
        #r.pid0.i=800
        #r.pid0.inputfilter=(0,0,0,0)
        # reset lockpoint
        #r.pid0.ival=0
        nT=-1
        for temperature in mivts_params['temperatures']:
            nT=nT+1
            log.info('Setting temperature to {} K'.format(temperature))
            self.temp_controller.ramp_rate(0)
            time.sleep(0.1)
            self.temp_controller.set_temperature(temperature)
            time.sleep(0.1)
            self.temp_controller.heater_range(mivts_params['heater_range'])
            time.sleep(50)
                
            with nidaqmx.Task() as read_task:
                for inst in DAQAnalogInputs.instances():
                    inst.close()
                daq_ai = DAQAnalogInputs('daq_ai', 'Dev2', samplerate, {'MAG': 4}, read_task,
                        samples_to_read=nsamples)
        
                with nidaqmx.Task() as write_task:   
                    write_task.ao_channels.add_ao_voltage_chan('Dev2/ao3')
                    nv0=-1
                
                    for v0 in v0_list:
                        nv0=nv0+1
                        
                        write_task.start()
                   #
                        write_task.write(v0,auto_start=True)
                        write_task.stop()
                        
                        time.sleep(0.1)
                        #print('Measure average current\r')
                        # measure average current
                        I_data=daq_ai.voltage()[0].T
                        
                        Iave_mat[nv0,nT]=np.mean(I_data)
                    ax.plot(v0_list,Iave_mat[:,nT])    
            
            daq_ai.close()
                     
        
        np.savetxt('Iave.dat',Iave_mat)
    def manual_plane(self, scan_params: Dict[str, Any]) -> Any:
    #     """
    #     Fine tune the scan plane by repeatedly scanning lines in the fast axis,
    #     keeping the slow axis fixed at zero, adjusting the x-slope, y-slope, and z
    #     parameters by entering keystrokes before each line scan

    #     Args:
    #         scan_params: Dict of scan parameters as defined
    #             in measuremnt configuration file.

    #     Returns:
    #         Tuple[qcodes.DataSet, plots.ScanPlot]: data, plot
    #             qcodes DataSet containing acquired arrays and metdata,
    #             and ScanPlot instance populated with acquired data.
    #     """
    #     #if not self.atto.plane_is_current:
    #     #    raise RuntimeError('Plane is not current. Aborting scan.')
        old_pos = self.scanner.position()
        
        daq_config = self.config['instruments']['daq']
        ao_channels = daq_config['channels']['analog_outputs']
        ao_channels = {}
        for ax in ['x', 'y', 'z']:
             ao_channels.update({ax: daq_config['channels']['analog_outputs'][ax]})
        ai_channels = daq_config['channels']['analog_inputs']
        meas_channels = scan_params['channels']
        channels = {}
        for ch in meas_channels:
             channels.update({ch: ai_channels[ch]})
        nchannels = len(channels.keys())

        daq_name = daq_config['name']
         #: DAQ AI sampling rate is divided amongst all active AI channels
        daq_rate = self.Q_(daq_config['rate']).to('Hz').magnitude / nchannels
        
        fast_ax = scan_params['fast_ax'].lower()
        slow_ax = 'x' if fast_ax == 'y' else 'y'
        
        line_duration = self.Q_(scan_params['range'][fast_ax]) / self.Q_(scan_params['scan_rate'])
        pts_per_line = int(daq_rate * line_duration.to('s').magnitude)
        pix_per_line = scan_params['scan_size'][fast_ax]
        pts_per_pix=int(pts_per_line/pix_per_line)
        lines_per_image = scan_params['scan_size'][slow_ax]
        center_line = int(np.floor(lines_per_image/2))
        plane = self.scanner.metadata['plane']
        height = self.Q_(scan_params['height']).to('V').magnitude
        scanner_constants = self.config['instruments']['scanner']['constants']
        scan_vectors = utils.make_scan_vectors(scan_params, scanner_constants, self.temp, self.ureg)
        scan_grids = utils.make_scan_grids(scan_vectors, slow_ax, fast_ax,
                                            pts_per_line, plane, height)
        utils.validate_scan_params(self.scanner.metadata, scan_params,
                                           scan_grids, self.temp, self.ureg, log)
        #pdb.set_trace()
        
        self.scanner.goto([scan_grids[axis][center_line][0] for axis in ['x', 'y', 'z']])
        #self.scanner.goto([scan_grids[axis][0][0] for axis in ['x', 'y', 'z']])
        # let the piezos relax before starting the scan
        time.sleep(10)
        while (1):
    
            new_input=input('xu,xd,yu,yd,zu,zd,r,q')
            if new_input == 'xu' :
                plane['x']=plane['x']+0.01
            if new_input == 'xd' :
               plane['x']=plane['x']-0.01
            if new_input == 'yu' :
               plane['y']=plane['y']+0.01
            if new_input == 'yd' :
               plane['y']=plane['y']-0.01
            if new_input == 'zu' :
               plane['z']=plane['z']+0.1
            if new_input == 'zd' :
               plane['z']=plane['z']-0.1
            if new_input == 'q':
               break    
            log.info('x={} y={} z={}'.format(plane['x'],plane['y'],plane['z']))                    
            scan_grids = utils.make_scan_grids(scan_vectors, slow_ax, fast_ax,
                                           pts_per_line, plane, height)
            utils.validate_scan_params(self.scanner.metadata, scan_params,
                                   scan_grids, self.temp, self.ureg, log)
            #this starts the built-in python debugger
            
            for npix in range (0,pix_per_line):
                npt=pts_per_pix*npix
                #pdb.set_trace()
                
                self.scanner.goto([scan_grids[axis][center_line][npt] for axis in ['x', 'y', 'z']],False,None,True)
            
            self.scanner.goto([scan_grids[axis][center_line][0] for axis in ['x', 'y', 'z']],False,None,True)        
               
        try:
            self.scanner.control_ao_task('stop')
            self.scanner.control_ao_task('close')
        except:
            pass
        self.scanner.goto([0, 0, 0])
    #         #self.CAP_lockin.amplitude(0.004)
    #         #self.SUSC_lockin.amplitude(0.004)
    #         log.info('Scan aborted by user. DataSet saved to {}.'.format(data.location))
        self.remove_component('daq_ai')
    #     utils.scan_to_mat_file(data, real_units=True)
        self.scanner.metadata['plane'].update({'x': plane['x'], 'y': plane['y'], 'z': plane['z']})
        return   
    
            