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

from instruments.lakeshore import Model_340, Model_331
from instruments.heater import EL320P
from utils import next_file_name
import matplotlib.pyplot as plt
from scipy import io
import time
import qcodes as qc

def BF4K_cooldown(fname=None, gpib340=16, sample_thermometer=True, gpib331=30,
                  stop_temp=1, dt=60, ts_fmt='%Y-%m-%d_%H:%M:%S'):
    """Logs fridge temperature (and optionally sample temperature) during a cooldown.
    """
    t0 = time.strftime(ts_fmt)
    if fname is None:
        fname = next_file_name('cooldown', 'mat')
    addr340 = 'GPIB0::{}::7::INSTR'.format(gpib340)
    ls340 = Model_340('ls340', addr340)
    time.sleep(0.1)
    T3K = ls340.B.temperature()
    time.sleep(0.1)
    T50K = ls340.A.temperature()
    Tsamp = '?'
    if sample_thermometer:
        addr331 = 'GPIB0::{}::7::INSTR'.format(gpib331)
        ls331 = Model_331('ls331', addr331,1)
        time.sleep(0.1)
        Tsamp = ls331.A.temperature()
    print('Cooldown started at {}.'.format(t0))
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K, Sample: {} K'.format(T50K, T3K, Tsamp))
    t = 0
    elapsed_time = [t]
    temp3K = [T3K]
    temp50K = [T50K]
    mdict = {'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K}
    if sample_thermometer:
        tempsamp = [Tsamp]
        mdict.update({'tempsamp': tempsamp})
    try:
        while T3K > stop_temp:
            for _ in range(int(dt)):
                time.sleep(1)
            t += dt
            T3K = ls340.B.temperature()
            time.sleep(0.1)
            T50K = ls340.A.temperature()
            elapsed_time.append(t)
            temp3K.append(T3K)
            temp50K.append(T50K)
            mdict.update({'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K})
            if sample_thermometer:
                time.sleep(0.1)
                Tsamp = ls331.A.temperature()
                tempsamp.append(Tsamp)
                mdict.update({'tempsamp': tempsamp})
            io.savemat(fname, mdict)
            plt.plot(elapsed_time, temp50K, 'r-', label='50K Plate')
            plt.plot(elapsed_time, temp3K, 'b-', label='3K Plate')
            if sample_thermometer:
                plt.plot(elapsed_time, tempsamp, 'k-', label='Sample')
            if t == dt:
                plt.legend(loc=0)
                plt.grid()
            plt.xlabel('Elapsed Time [s]')
            plt.ylabel('Temperature [K]')
            plt.title('BF4K Cooldown {}'.format(t0))
            plt.gcf().canvas.draw()
            plt.savefig(fname[:-3] + 'png')
        print('stop_temp reached at {}.'.format(time.strftime(ts_fmt)))
        plt.show()
    except KeyboardInterrupt:
        plt.show()
        print('Script interrupted by user at {}.'.format(time.strftime(ts_fmt)))
    qc.Instrument.close_all()
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K, Sample: {} K'.format(T50K, T3K, Tsamp))

def BF4K_warmup(fname=None, t_heater_off=290, t_stop_logging=295, heater_i=2, heater_v=30, dt=60, 
                gpib340=16, sample_thermometer=True, gpib331=30, heater_addr='ASRL3::INSTR',
                ts_fmt='%Y-%m-%d_%H:%M:%S'):
    """Applies (heater_i*heater_v) Watts to the 3 K plate and monitors temperature during a warmup.
    """
    if t_heater_off >= t_stop_logging:
        raise ValueError('t_heater_off must be less than t_stop_logging.')
    qc.Instrument.close_all()
    t0 = time.strftime(ts_fmt)
    if fname is None:
        fname = next_file_name('warmup', 'mat')
    addr340 = 'GPIB0::{}::7::INSTR'.format(gpib340)
    ls340 = Model_340('ls340', addr340)
    time.sleep(0.1)
    T3K = ls340.B.temperature()
    time.sleep(0.1)
    T50K = ls340.A.temperature()
    Tsamp = '?'
    if sample_thermometer:
        addr331 = 'GPIB0::{}::7::INSTR'.format(gpib331)
        ls331 = Model_331('ls331', addr331,1)
        time.sleep(0.1)
        Tsamp = ls331.A.temperature()
    t = 0
    elapsed_time = [t]
    temp3K = [T3K]
    temp50K = [T50K]
    mdict = {'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K}
    if sample_thermometer:
        tempsamp = [Tsamp]
        mdict.update({'tempsamp': tempsamp})
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K, Sample: {} K'.format(T50K, T3K, Tsamp))
    response = input('You are about to apply {} Watts to the 3 K plate.\nContinue with warmup? y/[n] '.format(heater_i * heater_v))
    if response.lower() != 'y':
        print('Warmup aborted.')
        for inst in Model_331.instances():
            inst.close()
        for inst in Model_340.instances():
            inst.close()
        return
    warmup_heater = EL320P('warmup_heater', heater_addr)
    err = warmup_heater.error()
    if err != 'OK':
        print('Heater error: {}. Turning heater off.'.format(err))
        warmup_heater.output('OFF')
        for inst in Model_331.instances():
            inst.close()
        for inst in Model_340.instances():
            inst.close()
        warmup_heater.close()
        return
    print('Warmup started at {}.'.format(t0))
    warmup_heater.voltage_set(heater_v)
    warmup_heater.current_set(heater_i)
    warmup_heater.output('ON')
    print('Applying {} Watts to 3 K plate.'.format(heater_i * heater_v))
    try:
        while T50K < t_stop_logging or T3K < t_stop_logging:
            err = warmup_heater.error()
            if err != 'OK':
                warmup_heater.output('OFF')
                raise RuntimeError('Heater error: {}. Turning heater off.'.format(err))
            for _ in range(int(dt)):
                time.sleep(1)
            t += dt
            T3K = ls340.B.temperature()
            time.sleep(0.1)
            T50K = ls340.A.temperature()
            elapsed_time.append(t)
            temp3K.append(T3K)
            temp50K.append(T50K)
            mdict.update({'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K})
            if sample_thermometer:
                time.sleep(0.1)
                Tsamp = ls331.A.temperature()
                tempsamp.append(Tsamp)
                mdict.update({'tempsamp': tempsamp})
            if warmup_heater.output() != 'OFF':
                if T3K > t_heater_off or T50K > t_heater_off:
                    print('t_heater_off reached at {}.'.format(time.strftime(ts_fmt)))
                    print('Turning heater off.')
                    warmup_heater.output('OFF')                
            io.savemat(fname, mdict)
            plt.plot(elapsed_time, temp50K, 'r-', label='50K Plate')
            plt.plot(elapsed_time, temp3K, 'b-', label='3K Plate')
            if sample_thermometer:
                plt.plot(elapsed_time, tempsamp, 'k-', label='Sample')
            if t == dt:
                plt.grid()
                plt.legend(loc=0)
            plt.xlabel('Elapsed Time [s]')
            plt.ylabel('Temperature [K]')
            plt.title('BF4K Warmup {}'.format(t0))
            plt.gcf().canvas.draw()
            plt.savefig(fname[:-3] + 'png')
        plt.show()
        print('t_stop_logging reached at {}.'.format(time.strftime(ts_fmt)))
    except KeyboardInterrupt:
        warmup_heater.output('OFF')
        io.savemat(fname, mdict)
        plt.show()
        print('Script interrupted by user at {}. Turning heater off.'.format(time.strftime(ts_fmt)))
    warmup_heater.output('OFF')
    for inst in Model_331.instances():
        inst.close()
    for inst in Model_340.instances():
        inst.close()
    for inst in EL320P.instances():
        inst.close()
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K, Sample: {} K'.format(T50K, T3K, Tsamp))