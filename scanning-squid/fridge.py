from instruments.lakeshore import Model_372
from utils import next_file_name
import matplotlib.pyplot as plt
from scipy import io
import time

def BF4K_cooldown(fname=None, ls_gpib=13, stop_temp=3, dt=60, ts_fmt='%Y-%m-%d_%H:%M:%S'):
    t0 = time.strftime(ts_fmt)
    if fname is None:
        fname = next_file_name('cooldown', 'mat')
    gpib = 'GPIB0::{}::7::INSTR'.format(ls_gpib)
    ls372 = Model_372('ls372', gpib)
    time.sleep(0.1)
    T3K = ls372.ch2.temperature()
    time.sleep(0.1)
    T50K = ls372.ch1.temperature()
    print('Cooldown started at {}.'.format(t0))
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K'.format(T50K, T3K))
    t = 0
    elapsed_time = [t]
    temp3K = [T3K]
    temp50K = [T50K]
    mdict = {'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K}
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
            mdict = {'time': elapsed_time, 'temp3K': temp3K, 'temp50K': temp50K}
            io.savemat(fname, mdict)
            try:
                plt.clear()
            except:
                pass
            plt.plot(elapsed_time, temp50K, 'ro-', label='50K Plate')
            plt.plot(elapsed_time, temp3K, 'bo-', label='3K Plate')
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
    print('Current temperature')
    print('-------------------')
    print('50K Plate: {} K, 3K Plate: {} K'.format(T50K, T3K))