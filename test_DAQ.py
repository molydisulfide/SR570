import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.fftpack import fft
import nidaqmx as ni

T_stop = 20
S_t = 0.5
N = int(T_stop/S_t)
timeout = time.time() + 60

task = ni.Task()
task.ai_channels.add_ai_voltage_chan("Dev3/ai0")


plt.close('all');
plt.ion();
plt.show();

fig, (ax1, ax2) = plt.subplots(2 ,1, figsize=(12,8))
fig.subplots_adjust(hspace = 1)


ax1.set_title('Current-Time Trace', fontsize = 15, fontname = 'Consolas')
ax1.set_ylabel('Current [nA]', fontsize = 12, fontname = 'Consolas')
ax1.set_xlabel('Time [x]', fontsize = 12, fontname = 'Consolas')
ax1.grid('on')
plt.xticks(fontname = 'Consolas')
plt.yticks(fontname = 'Consolas')
plt.setp(ax1.get_xticklabels(), visible = True, fontsize = 12)
plt.setp(ax1.get_yticklabels(), visible = True, fontsize = 12)


ax2.set_title('Live FFT', fontsize = 15, fontname = 'Consolas')
ax2.set_ylabel('Power', fontsize = 12, fontname = 'Consolas')
ax2.set_xlabel('Frequency [Hz]', fontsize = 12, fontname = 'Consolas')
ax2.grid('on')
ax2.set_xscale('log')
ax2.set_yscale('log')
plt.xticks(fontname = 'Consolas')
plt.yticks(fontname = 'Consolas')
plt.setp(ax2.get_xticklabels(), visible = True, fontsize = 12)
plt.setp(ax2.get_yticklabels(), visible = True, fontsize = 12)

data = np.array([]);

while True:
        time.sleep(0.5)
        raw = task.read()
        y = float(raw);
        
        data = np.append(data, y);
        
        #fft_data = np.fft.fft(data)
        #abs_fft_data = np.absolute(fft_data)
        #fft_freq = np.fft.fftfreq(len(abs_fft_data))
        
        ln1, = ax1.plot(data,'-', color = 'seagreen', lw = 2, alpha = 0.8)
        ln2, = ax2.plot(data,'-', color = 'steelblue', lw = 2, alpha = 0.8)
        
        plt.pause(0.001);
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
      
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        if time.time() > timeout:
            break
task.stop()