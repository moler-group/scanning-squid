from instruments.lakeshore import Model_372, Model_331
from utils import next_file_name
import matplotlib.pyplot as plt
from scipy import io
import time

def BF4K_cooldown(fname=None, gpib372=13, sample_thermometer=True, gpib331=30,
                  stop_temp=3, dt=60, ts_fmt='%Y-%m-%d_%H:%M:%S'):
    """Logs fridge temperature (and optionally sample temperature) during a cooldown.
    """
    t0 = time.strftime(ts_fmt)
    if fname is None:
        fname = next_file_name('cooldown', 'mat')
    addr372 = 'GPIB0::{}::7::INSTR'.format(gpib372)
    ls372 = Model_372('ls372', addr372)
    time.sleep(0.1)
    T3K = ls372.ch2.temperature()
    time.sleep(0.1)
    T50K = ls372.ch1.temperature()
    Tsamp = '?'
    if sample_thermometer:
        addr331 = 'GPIB0::{}::7::INSTR'.format(gpib331)
        ls331 = Model_331('ls331', addr331)
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
            T3K = ls372.ch2.temperature()
            time.sleep(0.1)
            T50K = ls372.ch1.temperature()
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
            try:
                plt.clear()
            except:
                pass
            plt.plot(elapsed_time, temp50K, 'ro-', label='50K Plate')
            plt.plot(elapsed_time, temp3K, 'bo-', label='3K Plate')
            if sample_thermometer:
                plt.plot(elapsed_time, tempsamp, 'ko-', label='Sample')
            if t == dt:
                plt.legend(loc=0)
            plt.xlabel('Elapsed Time [s]')
            plt.ylabel('Temperature [K]')
            plt.title('BF4K Cooldown {}'.format(t0))
            plt.gcf().canvas.draw()
            plt.savefig(fname[:-3] + 'png')
        print('stop_temp reached at {}.'.format(time.strftime(ts_fmt)))
    except KeyboardInterrupt:
        print('Script interrupted by user at {}.'.format(time.strftime(ts_fmt)))
    ls372.close()
    if sample_thermometer:
        ls331.close()
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K, Sample: {} K'.format(T50K, T3K, Tsamp))

def BF4K_warmup(t_heater_off=290, t_stop_logging=295, heater_i=2, heater_v=30, heater_addr='ASRL3::INSTR'):
    """Applies (heater_i*heater_v) Watts to the 3 K plate and monitors temperature during a warmup.
    """
    pass