# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:57:31 2021

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.fftpack import fft
import nidaqmx as ni

task = ni.Task()
task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
data = task.read()
print(data)

plt.ion()

fig, (ax1, ax2) = plt.subplots(2 ,1, figsize=(12,8))
fig.subplots_adjust(hspace = 1)

x_data, y_data = [], []
#ax1 = fig.add_subplot(1, 2)
ln1, = ax1.plot([], [],'-', color = 'seagreen', lw = 2, alpha = 0.8)
ax1.set_title('Current-Time Trace', fontsize = 15, fontname = 'Consolas')
ax1.set_ylabel('Current [nA]', fontsize = 12, fontname = 'Consolas')
ax1.set_xlabel('Time [x]', fontsize = 12, fontname = 'Consolas')
ax1.grid('on')
plt.xticks(fontname = 'Consolas')
plt.yticks(fontname = 'Consolas')
plt.setp(ax1.get_xticklabels(), visible = True, fontsize = 12)
plt.setp(ax1.get_yticklabels(), visible = True, fontsize = 12)

#ax2 = fig.add_subplot(2, 1)
ln2, = ax2.plot([], [],'-', color = 'steelblue', lw = 2, alpha = 0.8)
ax2.set_title('Live FFT', fontsize = 15, fontname = 'Consolas')
ax2.set_ylabel('Power', fontsize = 12, fontname = 'Consolas')
ax2.set_xlabel('Frequency [Hz]', fontsize = 12, fontname = 'Consolas')
ax2.grid('on')
ax2.set_xscale('log')
plt.xticks(fontname = 'Consolas')
plt.yticks(fontname = 'Consolas')
plt.setp(ax2.get_xticklabels(), visible = True, fontsize = 12)
plt.setp(ax2.get_yticklabels(), visible = True, fontsize = 12)
    
# =============================================================================
# SAMPLE_RATE = 20000  # Hertz
# DURATION = 10  # Seconds    
#   
# 
# def generate_sine_wave(freq, sample_rate, duration):
#     x_data = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x_data * freq
#     # 2pi because np.sin takes radians
#     y_data = np.sin((2 * np.pi) * frequencies)
#     return x_data, y_data     
# 
# x_data, y_data = generate_sine_wave(5, SAMPLE_RATE, DURATION)
# =============================================================================
    
while True:
    time.sleep(0.5)
    x_data = np.arange(10)
    y_data = np.random.random(10)
    
    ln1.set_xdata(x_data)
    ln1.set_ydata(y_data)
    
    fft_data = np.fft.fft(y_data)
    abs_fft_data = np.absolute(fft_data)
    fft_freq = np.fft.fftfreq(len(fft_data))
    
    ln2.set_xdata(fft_freq)
    ln2.set_ydata(abs_fft_data)
    
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    
    
    fig.canvas.draw()
    fig.canvas.flush_events()
