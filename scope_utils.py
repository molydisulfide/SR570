import pyvisa
import serial
import datetime
import pathlib
import time
import csv
import matplotlib
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

def write_sweep_figure(runtime, sweep, rc):
    ## Plots, saves an image of a voltage sweep, for comparison
    
    #start by setting up filename -- mirrors get_log_name() and get_log_dir()
    points_per_gate = int(rc['hold_time']*rc['sample_freq'])
    log_dir = get_log_dir(runtime)
    file_postfix = '_sweep.png'
    device_string = rc['dev_string']
    file_name = log_dir.joinpath(device_string+'_'+str(rc['Vac'])+'_mV_'+str(runtime.date().strftime('%Y%m%d'))+'_'+str(runtime.strftime("%H%M%S"))+file_postfix)
    time = []
    voltage = []
    for i, j in enumerate(sweep):
        for k in range(points_per_gate):
            time.append((len(time)+1)*1/rc['sample_freq'])
            voltage.append(j)
    fig = plt.figure(figsize=(14, 4))
    gs = matplotlib.gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(
        time, voltage, color="blue", label="V"
    )
    plt.savefig(file_name)
    fig.clear()
    plt.close(fig)
    return

## These utilities functions are used for startup
def get_run_time():
    return datetime.datetime.now()

def get_log_dir(runtime):
    date = runtime.date().strftime('%Y%m%d')
    write_dir = pathlib.Path(r'G:\GITHUB\ChimeraAcquisition\logfiles') # BUG If this ever runs on UNIX, this is gonna be a problem
    return write_dir.joinpath(date)

def get_log_name(runtime, rc):
    file_postfix = '_benchvue.csv'
    device_string = rc['dev_string']
    return device_string+'_'+str(rc['Vac'])+'_mV_'+str(runtime.date().strftime('%Y%m%d'))+'_'+str(runtime.strftime("%H%M%S"))+file_postfix

def get_filename(runtime, rc):
    filename = get_log_name(runtime, rc)
    log_dir = get_log_dir(runtime)
    return log_dir.joinpath(filename)

def verify_ootd(runtime, rc):
    log_dir = get_log_dir(runtime)
    file_name = pathlib.Path(str(log_dir) + '\OOTD.csv')  # BUG If this ever runs on UNIX, this \ is a problem.
    if file_name.exists():
        with open(file_name, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) > 0:
                    assert float(row[0]) == rc['Va']
    else:
        with open(file_name, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=' ', escapechar=' ', quoting=csv.QUOTE_NONE)
            writer.writerow([rc['Va']])
            csv_file.close()
    return

def set_up_scope4054(rc):
    # Get instrument VISAname
    myScope = 'USB0::2391::6066::MY56200356::INSTR'
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(myScope)
    scope.timeout = 10000
    scope.read_termination = '\n'
    scope.write_termination = '\n'
    IDN = scope.query("*IDN?")
    logging.info("Scope identifies as: "+IDN)
    all_scope_commands = ['SYSTem:PRESet', #or '*RST'
                          '*CLS',
                          ':AUToscale',
                          ':TIMebase:RANGe 0.001',
                          ':MEASure:VAVerage DISPlay, CHANnel' + rc['voltageChannel'],
                          ':MEASure:VAVerage DISPlay, CHANnel' + rc['currentChannel']]
    per_channel1_commands = ['PROBe 1',
                            'SCALe 250mV',
                            'OFFset 2.5V']  # we'll have to see if these play out OK
    per_channel2_commands = ['PROBe 1',
                            'SCALe 2V',
                            'OFFset 0V']
    for com in all_scope_commands:
        scope.write(com)
    for com in per_channel1_commands:
        scope.write("CHANnel4:"+com)
    for com in per_channel2_commands:
        scope.write("CHANnel2:"+com)
    return scope

def set_up_scope(rc):
    # Get instrument VISAname
    myScope = 'USB0::0x2A8D::0x1797::CN57096123::INSTR'
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(myScope)
    scope.timeout = 10000
    scope.read_termination = '\n'
    scope.write_termination = '\n'
    IDN = scope.query("*IDN?")
    logging.info("Scope identifies as: "+IDN)
    all_scope_commands = ['SYSTem:PRESet', #or '*RST'
                          '*CLS',
                          ':AUToscale',
                          ':TIMebase:RANGe 0.1',
                          ':MEASure:VAVerage DISPlay, CHANnel' + rc['voltageChannel'],
                          ':MEASure:VAVerage DISPlay, CHANnel' + rc['currentChannel']]
    per_channel1_commands = ['PROBe 1',
                            'SCALe 250mV',
                            'OFFset 2.5V']  # we'll have to see if these play out OK
    per_channel2_commands = ['PROBe 1',
                            'SCALe 2V',
                            'OFFset 0V']
    for com in all_scope_commands:
        scope.write(com)
    for com in per_channel1_commands:
        scope.write("CHANnel1:"+com)
    for com in per_channel2_commands:
        scope.write("CHANnel2:"+com)
    return scope

def find_amps(num_amps, com_ports):
    assert num_amps == 1, "Need to figure out autodetection of COM ports before you can ask for multiple amps"
    assert num_amps == len(com_ports), "You need to supply exactly one COM port per amp."
    amps = []
    for i in com_ports:
        amp = serial.Serial(i, ## COM1 or whatever
                            baudrate=9600,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_TWO)
        amps.append(amp)
    return(amps)
    
def init_amps_lownoise(amps, ootd, sens):
    init_commands = ['*RST',    # reset amp
                     'FLTT 4',  # filter type to 12 db/dec lowpass
                     'LFRQ 12', # filter low-pass freq to 12 for 30 kHz
                     'BLNK 0',  # blank front-end output
                     'ROLD',    # reset filter cap
                     'IOON 0',  # input offset curret disabled
                     'GNMD 0',  # gain mode low noise
                     'SUCM 0',  # sensitivity mode calibrated (as opposed to uncalibrated)
                     sens,      # sensitivity value to whatever is in the sense_LUT
                     'BSON 1',  # bias voltage active
                     'BSLV '+str(int(ootd)), # bias voltage value
                     ]
    for amp in amps:
        for com in init_commands:
            command = bytes(com+'\r\n', 'utf-8')
            amp.write(command)
            time.sleep(0.01)
            
def init_amps_highbandwidth(amps, ootd, sens):
    init_commands = ['*RST',
                     'FLTT 3',
                     'LFRQ 15',
                     'BLNK 0',
                     'ROLD',
                     'IOON 0',
                     'GNMD 1',
                     'SUCM 0',
                     sens,
                     'BSON 1',
                     'BSLV '+str(int(ootd)),]
    for amp in amps:
        for com in init_commands:
            command = bytes(com+'\r\n', 'utf-8')
            time.sleep(0.05)
            amp.write(command)
            time.sleep(0.05)

## These utility functions are used during the run

def vg_setrel(ootd, vac, vg):  # BUG check if we still need this.
    '''
    Input: ootd, the current vac and a desired vg value relative to that vac
    ("I have ootd=2550 and am at vac=500, and I want the gate to be at 200 above the midpoint")
    Returns: the value to which to set the 570 voltage to get that gate voltage
    '''
    return vg + ootd + vac/2  # BUG: the fact that the test suite passes does not mean that the math may not be wrong -- we should check!
            
def amp_error(setpoint, measurement):
    return setpoint-measurement
            
def amp_check(setpoint, measurement, rc):
    if abs(amp_error(setpoint, measurement)) < rc['allowed_amp_error']:
        return True
    else:
        return False
    
def get_sweep(V_ac, V_g_max, steps, ootd, func_name):
    the_sweep = func_name(V_ac, V_g_max, steps, ootd)
    logging.info("The sweep is: "+str(the_sweep))
    return the_sweep
    
def sweep_bidirectional(V_ac, V_ag_max, steps, Va):
    '''
    Given a max allowable Vag, step count per side and offset, returns the gate voltages V_AG
    '''
    Vag = []
    step_size = V_ag_max/float(steps)
    offset = Va-V_ac/2 # that's the midpoint of V_ac
    for i in range(steps):
        Vag.append(offset+step_size*i)
    for i in range(steps):
        Vag.append(offset+V_ag_max-step_size*i)
    for i in range(steps):
        Vag.append(offset-step_size*i)
    for i in range(steps):
        Vag.append(offset-V_ag_max+step_size*i)
    Vag.append(float(offset))
    return Vag

def sweep_oneway(V_ac, V_ag_max, steps, Va):
    '''
    Given a max allowable vg, step count per side and offset, returns the gate voltages relative to true ground for a monotonic climb.
    '''
    Vag = []
    step_size = V_ag_max/float(steps)
    offset = Va-V_ac/2 # midpoint of Vac
    for i in range(2*steps):
        Vag.append(offset-V_ag_max+step_size*i)
    Vag.append(float(offset))
    return Vag

def sweep_single(V_ac, V_ag_max, steps, Va):
    '''
    Given a max allowable vg, step count per side and offset, returns the gate voltages relative to true ground for a single step, and return..
    '''
    assert(steps == 1)
    Vag = []
    step_size = V_ag_max
    offset = Va-V_ac/2
    Vag.append(offset+step_size)
    Vag.append(float(offset))
    return Vag

def read_VAVerage(scope, channel):
    result=scope.query(":MEASure:VAVerage? DISPlay, CHANnel" + channel )
    return float(result)

def read_channel_1(scope):
    result=scope.query(":MEASure:VAVerage? DISPlay, CHANnel1")
    return float(result)

def read_channel_2(scope):
    result=scope.query(":MEASure:VAVerage? DISPlay, CHANnel2")
    return float(result)
## BUG?  This does not know that source2 is a current, and the current handler in the analysis suite handles that.

def amp_set_gate(amp, gate):
    assert type(gate) == float
    logging.info("\n\t\t\t\tSetting gate to \t {s0:5.2f}  mV\n".format(s0=gate))
    command = 'BSLV '+str(int(gate))+'\r\n'
    amp.write(bytes(command, 'utf-8'))

## These utility functions are used when closing things down.
    
def close_amps(amps):
    for amp in amps:
        amp.close()
    for amp in amps:
        assert not amp.is_open  # if COM1 stays open, restart Spyder...
