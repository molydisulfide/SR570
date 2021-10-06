## Run Constants
## TODO -- do we care about not having these hardcoded?

rc = {}

## these you change on every run

rc['dev_string'] = 'modelCell_0MKCl_pindowngatesweep2noscope'
rc['Vac'] = 600 # millivolt

## these you change about once a day.

rc['sens'] = '1000' # 10000 nA/V is 10 uA/V, which goes to 200k with 2pA/sqrt(Hz) noise  at 100kHz, this is 200 pA
# however, that one pushes too much input current with the head open, so stay lower there.
rc['Va'] = 2350 # offset of the day, millivolt

## these you change at will

## OK, now caution -- the front and back crops are hardcoded in Schnellstapel, so if your interval ius too short here, it will throw an assertion error

rc['hold_time'] = 5 # sec for how much you hold the gate voltage
rc['steps'] = 2 # number of steps in gate sweep
rc['max_gate'] = abs(rc['Vac']/2)-50 # max differential gate sweep voltage, mV
#rc['max_gate'] = 300
rc['sample_freq'] = 1.5 # hertz, for sampling of the gate current, after downconversion
rc['allowed_amp_error'] = 20 #mV of allowable deviation between the 570 setpoint and the scope measurement

### Below here, we have derived run constants

rc['run_length'] = rc['hold_time']*(2*rc['steps']+1)

rc['voltageChannel']='1'
rc['currentChannel']='2'

rc['niSamplingRate'] = 50_000  #Hz
# max tested 200k Hz seems working without problem ONLY FOR 1 device. 
# Since we use 2 DAQ devices the max samplerate is 100kHz. 
# If you want to improve, log the voltage and current on independent threads. 
rc['niPrintInterval'] = 2 #sec
# The NI and scope sampling are completely in parallel, so both files will be written.
rc['ni_sample_freq'] = 25_000 # Hz, target frequency for Benchvue output after conversion.

import ni_utils_2devices as ni_utils
import matplotlib.pyplot as plt
import numpy as np
import time


sens_lut = {
            '0.001': 'SENS 0',
            '0.002': 'SENS 1',
            '0.005': 'SENS 2',
            '0.010': 'SENS 3',
            '0.020': 'SENS 4',
            '0.050': 'SENS 5',
            '0.100': 'SENS 6',
            '0.200': 'SENS 7',
            '0.500': 'SENS 8',
            '1': 'SENS 9',
            '2': 'SENS 10',
            '5': 'SENS 11',
            '10': 'SENS 12',
            '20': 'SENS 13',
            '50': 'SENS 14',
            '100': 'SENS 15',
            '200': 'SENS 16',
            '500': 'SENS 17',
            '1000': 'SENS 18',
            '2000': 'SENS 19',
            '5000': 'SENS 20',
            '10000': 'SENS 21',
            '20000': 'SENS 22',
            '50000': 'SENS 23',
            '100000': 'SENS 24',
            '200000': 'SENS 25',
            '500000': 'SENS 26',
            '1000000': 'SENS 27'
        }

assert rc['sens'] in sens_lut

#check that integral points per gate sweep
assert rc['hold_time']*(2*rc['steps']+1) - rc['run_length'] == 0, 'You need to spec more time for that step setting.'

import time
import datetime
import csv
import serial
import threading
import sys
import logging
logging.basicConfig(level=logging.INFO)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--init",
                    action='store_true',
                    help="Initialize the 570.  DONT DO THIS IF THE PIN IS DOWN!")
parser.add_argument("--single",
                    action='store_true',
                    help="Run a single measurement.")
parser.add_argument("--run",
                    action='store_true',
                    help="Run a measurement.")
parser.add_argument("-v", "--voltage", type=float, help="Set explicit gate voltage.")
#parser.add_argument("-t", "--time", type=float, help="Write time.")
args = parser.parse_args()

import scope_utils

def read_a_pair(scope, value, rc):
    result1=1000.0*scope_utils.read_VAVerage(scope,rc['voltageChannel'])
    print('Is the gate voltage within the allowed error:', str(scope_utils.amp_check(value, result1, rc)))
    print("Measure raw V_G: %s" % result1)
    result2=float(rc['sens'])*scope_utils.read_VAVerage(scope,rc['currentChannel'])
    print("Measure I_G: %s" % result2)
    return result1, result2

def handle_serial_fail(amps):
    logging.info("You just recovered from a COM port failure.")
    scope_utils.close_amps(amps)
    new_amps = scope_utils.find_amps(1, ['COM1']) ## TODO -- this assumes we only have one device on one COM port.
    return new_amps


def run_init(amps, sens):
    logging.info('You have asked to initialize the SR570s.')
    scope_utils.init_amps_lownoise(amps, rc['Va'], sens) #maybe _highbandwisth
    scope_utils.close_amps(amps)
    logging.info('Init finished.')

def run_single(args, amps, scope, rc):
    logging.info('You are in one-shot mode, and will not record a file.')
    if not args.voltage:
        logging.info("You asked for a single run, but didn't specify a gate.  Setting gate to midpoint.")
        v = 0
    else:
        logging.info("Setting gate to "+str(args.voltage))
        v = args.voltage
    sweep = rc['Va']-rc['Vac']/2 + v #BUG -- check here -- the intent is to get args.v = 0 to be GATE OFF
    scope_utils.amp_set_gate(amps[0], sweep)
    logging.info('Voltage setting without file finished, now to check it.')
    chan1, chan2 = read_a_pair(scope, sweep, rc)
    
## The main goodies run here,
if __name__ == '__main__':
    scope = scope_utils.set_up_scope(rc)
    #scope = scope_utils.set_up_scope(rc)
    #BUG? OK, this is dumb, but the main loop does not run unless you probe each scope channel first with a timeout error on the read.  Not sure if the initial read is way slow or something.
    logging.info('Startup gimmicks.')
    result1=scope.query(":MEASure:VAVerage? DISPlay, CHANnel" + rc['voltageChannel'])
    logging.info("Measure source1: %s" % result1)
    result2=scope.query(":MEASure:VAVerage? DISPlay, CHANnel" + rc['currentChannel'])
    logging.info("Measure source2: %s" % result2)
    logging.info('Startup gimmicks complete.')
    try:
        amps = scope_utils.find_amps(1, ['COM1'])
    except serial.serialutil.SerialException:
        handle_serial_fail(amps)

    # initialize NI
    niFile = scope_utils.get_filename(datetime.datetime.now(), rc)
    niFile = niFile.parent / (niFile.stem + '_NIlogger.csv')
    
    try:
        d, t, r = ni_utils.initialize_NI(ni_utils.get_NI_config(rc))
        f = ni_utils.NI_logFile(niFile, t.currentTask.config)
        p = ni_utils.NI_single_reader_wrapper(t)
    except Exception as e:
        logging.warning(e)
        logging.warning('Failed creating nidaqtask.  Resetting instrument and retrying.')
        d = ni_utils.NI_device(ni_utils.get_NI_config(rc))
        d.reset_NI()
        d, t, r = ni_utils.initialize_NI(ni_utils.get_NI_config(rc))
        f = ni_utils.NI_logFile(niFile, t.currentTask.config)
        p = ni_utils.NI_single_reader_wrapper(t)

    stopLogging = threading.Event()
    printOut = threading.Event()
    
    loggerThread = threading.Thread(
        target = ni_utils.log_data_NI, 
        args= (r,f,stopLogging))

    printerThread = threading.Thread(
            target = ni_utils.display_data_NI, 
            args = (p,stopLogging,printOut,rc['niPrintInterval']), 
            name = 'NI_printer_thread')
    
    plottingThread = threading.Thread(
            target = ni_utils.plot_data_NI,
            args = (r, stopLogging))
    
    sens = sens_lut[rc['sens']]

    try:
        if args.init:
            logging.info(' Setting up SR570 \t...')
            run_init(amps, sens)
            logging.info(' Setup of SR570 \t DONE')
            time.sleep(2)
        elif args.single:
            loggerThread.start()
            f.set_start_time(datetime.datetime.now())
            printerThread.start()
            try: 
                run_single(args, amps, scope, rc)
            except serial.serialutil.SerialException:
                amps = handle_serial_fail(amps)
                run_single(args, amps, scope, rc)
    
    
    except KeyboardInterrupt:
        logging.info('  interrupted the programm via key board - will try to shutdown cleanly')
        stopLogging.set()  # inform the child thread that it should exit
        t.unload()
        scope_utils.close_amps(amps)
        sys.exit(1)

    stopLogging.set()
    
    ## Wrap things up
    scope_utils.close_amps(amps)

    # close ni logger
    t.unload()
    logging.info('NI logging should be finished correctly')
    
    # BUG Maybe you might need a t.task.close() here, but figure out what the difference is here.
    try:
        t.task.close()
    except:
        logging.info('NI logging should be finished correctly')

    logging.info('Run finished.')
# =============================================================================

# =============================================================================
   
# =============================================================================
