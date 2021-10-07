#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nidaqmx
import numpy as np
from nidaqmx import constants
from nidaqmx import stream_readers
import threading
import time
import csv
import datetime
import pathlib
import matplotlib.pyplot as plt
from scipy.signal import decimate, sosfiltfilt, firwin,  bessel
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

#NOTE that this rc{} has nothing to do with the the rc in scope_control, and is 
# basically a way to run this thing, in main, which we should do ony to test either way.
rc = {}
rc['niSamplingRate'] = 50000
rc['sens'] = '100' # nA/V
timeout = time.time() + 60

class NI_config:
    def __init__(self, channelIdent, sampleType, LBUB, samplingRate, samplingMode, terminalConfiguration, scaleParameter = 1.0):
        self.channelIdent = channelIdent
        self.sampleType = sampleType
        self.LBUB = LBUB
        self.samplingRate = samplingRate
        self.samplingMode = samplingMode
        self.bufferSize = samplingRate
        self.timeout = 1.5 * self.bufferSize / self.samplingRate
        self.terminalConfiguration = terminalConfiguration
        self.scaleParameter = scaleParameter

class NI_device:
    def __init__(self, *config):
        # config = list(config[0])
        self.device1 = nidaqmx.system.device.Device(name=config[0][0].channelIdent.split('/')[0])
        self.device2 = nidaqmx.system.device.Device(name=config[0][1].channelIdent.split('/')[0])
    def reset_NI(self):
        self.device1.reset_device()
        self.device2.reset_device()
        return

class NI_task_wrapper:
    def __init__(self, config):
        self.voltageTask=NI_task(config[0])
        self.currentTask=NI_task(config[1])
    def unload(self):
        self.voltageTask.unload()
        self.currentTask.unload()
        del self.voltageTask, self.currentTask,

class NI_task:
    def __init__(self, *config):
        config = list(config)
        self.config = config[0]
        self.task = nidaqmx.Task()
        self.scaleParameter = []
        for conf in config:
            self.create_channel(conf)
            self.scaleParameter.append(conf.scaleParameter)
        self.create_timer()
        #self.task.in_stream.offset = 0
        self.task.in_stream.over_write=constants.OverwriteMode.DO_NOT_OVERWRITE_UNREAD_SAMPLES

    def create_channel(self,config):
        if config.sampleType =='voltage':
            self.task.ai_channels.add_ai_voltage_chan(
                    physical_channel=config.channelIdent, 
                    min_val=config.LBUB[0], 
                    max_val=config.LBUB[1], 
                    terminal_config = config.terminalConfiguration
                    )
            
        elif config.sampleType =='current':
            self.task.ai_channels.add_ai_current_chan(
                    physical_channel=config.channelIdent, 
                    min_val=config.LBUB[0], 
                    max_val=config.LBUB[1], 
                    terminal_config = config.terminalConfiguration
                    )

    def create_timer(self):
        self.task.timing.cfg_samp_clk_timing(
            rate=self.config.samplingRate, 
            sample_mode=self.config.samplingMode, 
            samps_per_chan=self.config.bufferSize * 10
            )
        ## BUG - Clock is not synchroized accross devices - there is an offset of a couple of samples between both channels
        
    def unload(self):
        self.task.stop()
        self.task.close()
        del self.task

class NI_reader_wrapper:
    def __init__(self, NI_task_wrapper):
        self.voltageReader = NI_reader(NI_task_wrapper.voltageTask)
        self.currentReader = NI_reader(NI_task_wrapper.currentTask)
        self.columns = self.voltageReader.buffer.shape[0] + self.currentReader.buffer.shape[0]
        self.rows = self.voltageReader.buffer.shape[1]
        self.data = np.zeros((self.rows,self.columns), dtype=np.float64)

    def read(self):
        self.voltageReader.read()
        self.currentReader.read()

    def get_data(self):
        self.data[:,0:self.voltageReader.buffer.shape[0]] = self.voltageReader.get_data()
        self.data[:,self.columns-self.currentReader.buffer.shape[0]:] = self.currentReader.get_data()
        return self.data

class NI_reader:
    def __init__(self, NI_task):
        self.reader = stream_readers.AnalogMultiChannelReader(NI_task.task.in_stream)
        self.numSamples = NI_task.config.bufferSize
        self.timeout = NI_task.config.timeout
        self.buffer = np.zeros((NI_task.task.number_of_channels, NI_task.config.bufferSize), dtype=np.float64)
        self.scaleParameter = NI_task.scaleParameter
        
    def read(self):
        self.reader.read_many_sample(self.buffer, self.numSamples, timeout=self.timeout)

    def get_data(self):
        data = self.buffer.T.astype(np.float32)
        data = data * self.scaleParameter
        return data[:]

class NI_single_reader_wrapper:
    def __init__(self, NI_task_wrapper):
        self.voltageReader = NI_single_reader(NI_task_wrapper.voltageTask)
        self.currentReader = NI_single_reader(NI_task_wrapper.currentTask)

    def read(self):
        self.voltageReader.read()
        self.currentReader.read()

    def get_data(self):
        x = self.voltageReader.buffer.shape[0] + self.currentReader.buffer.shape[0]
        data = np.zeros(x, dtype=np.float64)
        data[0:self.voltageReader.buffer.shape[0]] = self.voltageReader.get_data()
        data[x-self.currentReader.buffer.shape[0]:] = self.currentReader.get_data()
        return data

class NI_single_reader:
    def __init__(self, NI_task):
        NI_task.task.in_stream.ReadRelativeTo=constants.ReadRelativeTo.MOST_RECENT_SAMPLE
        self.reader = stream_readers.AnalogMultiChannelReader(NI_task.task.in_stream)#, relative_to=constants.ReadRelativeTo.MOST_RECENT_SAMPLE))
        self.numSamples = 1
        self.timeout = NI_task.config.timeout
        self.buffer = np.zeros(NI_task.task.number_of_channels, dtype=np.float64)
        self.scaleParameter = NI_task.scaleParameter
        
    def read(self):
        self.reader.read_one_sample(self.buffer, timeout=0.1)#, offset = -1, relative_to=constants.ReadRelativeTo.MOST_RECENT_SAMPLE)
# Most Recent Sample 
    def get_data(self):
        data = self.buffer.T
        data = data * self.scaleParameter
        return data[:]

class NI_logFile:
    def __init__(self, fileName, NI_config):
        self.fileName = fileName
        with open(self.fileName, 'a') as csv_file:
            self.writer = csv.writer(csv_file)
            self.writer.writerow(['voltage','current'])
        self.startTime = datetime.datetime.now()
        self.config = NI_config
    
    def set_start_time(self, timestamp):
        self.startTime = timestamp
                    
    def file_writer(self, buffer):
        with open(self.fileName, 'a') as csv_file:
            self.writer = csv.writer(csv_file)
            self.writer.writerows(buffer)
            

    def FIR_lowpass(self, newSampleRate):
        numtaps = int(np.floor(2.0/3.0*np.log10(1.0/10.0/1.0e-4/1.0e-3)*self.config.samplingRate/newSampleRate*2.0))
        taps = firwin(numtaps, newSampleRate, window = 'hamming', pass_zero = 1, fs = self.config.samplingRate)
        a = [1.0]
        return taps, a

    def bessel_lowpass(self, newSampleRate, order = 8):
        nyq = 0.5 * self.config.samplingRate
        high = newSampleRate / nyq
        sos = bessel(order, high, output='sos')
        return sos

def initialize_NI(config):
    d = NI_device(config)
    t = NI_task_wrapper(config)
    r = NI_reader_wrapper(t)
    return d, t, r

def display_data_NI(NI_reader_wrapper, stopLogging, printOut, printInterval):
    while not stopLogging.isSet():
        NI_reader_wrapper.read()
        data = NI_reader_wrapper.get_data()
        if not printOut.wait(printInterval):
            logging.info('\t\t\t\tNI: Voltage: {s0:5.2f} mV'.format(s0=data[0]))
            logging.info('\t\t\t\tNI: Current: {s0:8.3f} nA'.format(s0=data[1])) 
            
    logging.info('\t\t\t\tPrinting done because end of stop logging flag')

def log_data_NI(NI_reader_wrapper, NI_logFile, stopLogging):
    while not stopLogging.isSet():  # This thing means that stopLogging.set() can only be called *ONCE* before you have to tear down and rebuild some instance logTask 
        NI_reader_wrapper.read()
        data = NI_reader_wrapper.get_data()
        NI_logFile.file_writer(data)
    logging.info('\t\t\t\tLogging done because end of stop logging flag.')
    
           
def get_NI_config(rc):
    config1 = NI_config(
        channelIdent ="Dev3/ai0", 
        sampleType = 'voltage', 
        LBUB = [-5,5], 
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS, 
        terminalConfiguration = constants.TerminalConfiguration.DIFFERENTIAL,
        scaleParameter = 1000.0)

    config2 = NI_config(
        channelIdent ="Dev4/ai0", 
        sampleType = 'voltage', 
        LBUB = [-0.1,0.1],
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS,
        terminalConfiguration = constants.TerminalConfiguration.RSE,
        scaleParameter = np.float(rc['sens']))
    
    return config1, config2

printInterval = 1
                
def main():
    fileName = pathlib.Path.cwd()
    runtime = datetime.datetime.now()
    fileStr = 'I-t_Trace_' + str(runtime.date().strftime('%d%m%Y'))+'_'+str(runtime.strftime("%H%M")) + '.csv'
    fileName =  fileName / fileStr
        
    try:
        d, t, r = initialize_NI(get_NI_config(rc))
        f = NI_logFile(fileName, t.currentTask.config)
        p = NI_single_reader_wrapper(t)
    except Exception as e:
        logging.warning('Failed creating nidaqtask. Resetting instrument and retry.')
        logging.warning(e)
        d = NI_device(get_NI_config(rc))
        d.reset_NI()
        d,t,r = initialize_NI(get_NI_config(rc))
        f = NI_logFile(fileName, t.currentTask.config)
        p = NI_single_reader_wrapper(t)

    stopLogging = threading.Event()
    printOut = threading.Event()

    loggerProcess = threading.Thread(
        target = log_data_NI, 
        args = (r,f,stopLogging), 
        name = 'NI_logger_thread',
        daemon = True)
    loggerProcess.start()
    f.set_start_time(datetime.datetime.now())

    printerProcess = threading.Thread(
        target = display_data_NI, 
        args = (p,stopLogging,printOut,printInterval), 
        name = 'NI_printer_thread',
        daemon = True)
    printerProcess.start()

    
    time.sleep(10)

    stopLogging.set()
    #time.sleep(t.config.bufferSize / t.config.samplingRate * 2.0 )
    t.unload()
   

if __name__ == "main":
    main()
    