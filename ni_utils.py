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
from scipy.signal import decimate, sosfiltfilt, firwin,  bessel
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

#NOTE that this rc{} has nothing to do with the the rc in scope_control, and is 
# basically a way to run this thing, in main, which we should do ony to test either way.
rc = {}
rc['niSamplingRate'] = 50000
rc['sens'] = '100' # nA/V

class NI_config:
    def __init__(self, channelIdent, sampleType, LBUB, samplingRate, samplingMode, scaleParameter = 1.0):
        self.channelIdent = channelIdent
        self.sampleType = sampleType
        self.LBUB = LBUB
        self.samplingRate = samplingRate
        self.samplingMode = samplingMode
        self.bufferSize = samplingRate // 2
        self.timeout = 1.5 * self.bufferSize / self.samplingRate
        self.scaleParameter = scaleParameter

class NI_device:
    def __init__(self, *config):
        config = list(config[0])
        self.device = nidaqmx.system.device.Device(name=config[0].channelIdent.split('/')[0])
    def reset_NI(self):
        self.device.reset_device()
        return

class NI_task:
    def __init__(self, config):
        config = list(config[0])
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
                    # min_val=config.LBUB[0], 
                    # max_val=config.LBUB[1], 
                    # terminal_config = constants.TerminalConfiguration.RSE
                    )
            
        elif config.sampleType =='current':
            self.task.ai_channels.add_ai_current_chan(
                    physical_channel=config.channelIdent, 
                    min_val=config.LBUB[0], 
                    max_val=config.LBUB[1], 
                    terminal_config = constants.TerminalConfiguration.RSE
                    )

    def create_timer(self):
        self.task.timing.cfg_samp_clk_timing(
            rate=self.config.samplingRate, 
            sample_mode=self.config.samplingMode, 
            samps_per_chan=self.config.bufferSize * 100
            )
        
    def unload(self):
        self.task.stop()
        self.task.close()
        del self.task

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
        return data[:,:2]

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
        return data[:2]

class NI_logFile:
    def __init__(self,fileName, NI_config):
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
            
    def make_benchvue(self, newSampleRate):
        fullData = pd.read_csv(self.fileName).values#, dtype = np.float32, skip_header=1, encoding = 'latin1')
        fullTime = 1.0 / self.config.samplingRate * len(fullData)
        if newSampleRate < self.config.samplingRate:
            fullData = self.filter_benchvue(fullData, newSampleRate)
            fullData = self.downsample_benchvue(fullData, newSampleRate)
            dt = fullTime / len(fullData)
        else:
            dt = 1.0 / self.config.samplingRate
        tS = datetime.datetime.timestamp(self.startTime)
        index = np.arange(0,len(fullData),dtype = int)
        timeStamp = [datetime.datetime.fromtimestamp(tS + dt*i) for i in index]
        benchvueFile = self.fileName.parent / (self.fileName.stem + 'benchvue.csv')
        with open(benchvueFile, 'w') as csv_file:
            self.writer = csv.writer(csv_file, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
            for i in index:
                self.writer.writerow([str(i)+',,,'+str(timeStamp[i]) + ','+'{:.2f}'.format(fullData[i,0])+','+'{:.3f}'.format(fullData[i,1])])
        del fullData, index, timeStamp

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

    def filter_benchvue(self, fullData, newSampleRate):
        # b, a = self.FIR_lowpass(newSampleRate//2) too memory intensiv on the computer!
        # return np.array(
        #         [sosfiltfilt(b, a,fullData[:,0], padlen = 3 * max(len(a), len(b)), padtype = "even"),
        #         sosfiltfilt(b, a,fullData[:,1], padlen = 3 * max(len(a), len(b)), padtype = "even")]).T
        sos = self.bessel_lowpass(newSampleRate//2, order = 8)
        return np.array(
                [sosfiltfilt(sos, fullData[:,0]),
                sosfiltfilt(sos, fullData[:,1])]).T



    def downsample_benchvue(self, fullData, newSampleRate):
        downScale = int(np.floor(self.config.samplingRate / newSampleRate))
        return np.array(
                [decimate(fullData[:,0], downScale, n=20, ftype="fir", zero_phase=True),
                decimate(fullData[:,1], downScale, n=20, ftype="fir", zero_phase=True)]).T

def initialize_NI(*config):
    d = NI_device(config[0])
    t = NI_task(config)
    #t.task.start()
    r = NI_reader(t)
    return d, t, r

def display_data_NI(NI_reader, stopLogging, printOut, printInterval):
    while not stopLogging.isSet():
        NI_reader.read()
        data = NI_reader.get_data()
        if not printOut.wait(printInterval):
            logging.info('\t\t\t\tNI: Measure V_G: {s0:5.2f} mV'.format(s0=data[0]))
            logging.info('\t\t\t\tNI: Measure I_G: {s0:8.3f} nA'.format(s0=data[1]))        
    logging.info('Printing done because end of stop logging flag')

def log_data_NI(NI_reader, NI_logFile, stopLogging):
    while not stopLogging.isSet():
        NI_reader.read()
        data = NI_reader.get_data()
        NI_logFile.file_writer(data)
    logging.info('Logging done because end of stop logging flag.')

def get_NI_config(rc):
    config1 = NI_config(
        channelIdent ="Dev3/ai0", 
        sampleType = 'voltage', 
        LBUB = [-5,5], 
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS, 
        scaleParameter = 1000.0)

    # add channel to reduce ghosting
    config2 = NI_config(
        channelIdent ="Dev3/ai1", 
        sampleType = 'voltage', 
        LBUB = [-2,2], 
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS, 
        scaleParameter = 1.0)

    config3 = NI_config(
        channelIdent ="Dev3/ai2", 
        sampleType = 'voltage', 
        LBUB = [-2,2], 
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS, 
        scaleParameter = np.float(rc['sens']))
    
    # add channel to reduce ghosting
    config4 = NI_config(
        channelIdent ="Dev3/ai3", 
        sampleType = 'voltage', 
        LBUB = [-5,5], 
        samplingRate = rc['niSamplingRate'], 
        samplingMode = constants.AcquisitionType.CONTINUOUS, 
        scaleParameter = 1.0)
    return config1, config2, config3, config4

printInterval = 1
                
def main():
    fileName = pathlib.Path.cwd()
    runtime = datetime.datetime.now()
    fileStr = 'logfile_' + str(runtime.date().strftime('%Y%m%d'))+'_'+str(runtime.strftime("%H%M%S")) + '.csv'
    fileName =  fileName / fileStr
        
    try:
        d,t,r = initialize_NI(get_NI_config(rc))
        f = NI_logFile(fileName, t.config)
        p = NI_single_reader(t)
    except Exception as e:
        logging.warning('Failed creating nidaqtask. Resetting instrument and retry.')
        logging.warning(e)
        d = NI_device(get_NI_config(rc))
        d.reset_NI()
        d,t,r = initialize_NI(get_NI_config(rc))
        f = NI_logFile(fileName, t.config)
        p = NI_single_reader(t)

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
    
    time.sleep(30)

    stopLogging.set()
    #time.sleep(t.config.bufferSize / t.config.samplingRate * 2.0 )
    t.unload()
    f.make_benchvue(100)
    logging.info('  Benchvue conversion successful.')
    try:
        t.task.close()
    except:
        logging.info('Should be finished correctly.')

if __name__ == "main":
    main()
    