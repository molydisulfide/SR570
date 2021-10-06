# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:01:11 2020

The MultipleDataPlotter class interprets data coming off of one or more MOANA chips and plots it.

@author: Kevin Renehan
"""

import numpy as np
import matplotlib.pyplot as plt
import time
# from easygui import fileopenbox
# from easygui import diropenbox
from sys import exit
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from os import path

# ===========================================================
# Data Plotter class for parsing, plotting, and saving MOANA data
# ===========================================================
class MultipleDataPlotter:
    
    # Class options
    averaging = False
    cut_large_patterns = False
    save_figures = False
    report_pattern_totals = False
    show_plots = True
    raw_plotting = False
    persistent_plotting = False
    accumulated_plotting = False
    show_mean = False
    show_peak = False
    show_coarse_fine = False
    plot_logarithmic = False
    show_regression = False
    fix_y_max = False
    fast_mode = False
    show_plot_info = False
    
    # Number of chips
    __number_of_chips = 1
    
    # Frame variables
    __number_of_frames = 1
    __meas_per_patt = 1
    __patt_per_frame = 1
    
    # Class variables
    __number_of_bins = 150
    __bin_size = 12
    __transfer_size = 1824
    
    # Subtractor value
    __subtractor_value = 0
    
    # Persistence depth
    __persistence_depth = 1
    
    # VCSEL
    __vcsel_setting = 0
    __vcsel_setting_set = False
    __vcsel_delays = [0.12, 0.24, 0.36, 0.48]
    __vcsel_latency = 0.72
    
    # SPADs
    __spad_rise_time = 0
    __tdc_resolution = 65e-3
        
    # The current capture number
    __current_capture = 0 
    
    # Linear regression
    __rsquared = 0.0
    __slope = 0.0
    __intercept = 0.0
    
    # Axis limits
    __ymax = 0.0
    
    # Plot status
    __plot_figure_spawned = False
    __plot_legend_spawned = False
    
    # Text handles
    __text_handle = None
    

    # ===========================================================
    # Constructor
    # ===========================================================
    def __init__(self, number_of_chips, meas_per_patt, patt_per_frame, number_of_frames, period, time_limits=[-1, -1], persistence_depth=1):
        
        # Initialize parameters
        self.__number_of_chips                  = number_of_chips
        self.__meas_per_patt                    = meas_per_patt
        self.__patt_per_frame                   = patt_per_frame
        self.__number_of_frames                 = number_of_frames
        self.__period                           = period
        self.__timestep                         = 5e-3
        self.__persistence_depth                = persistence_depth
        
        # Time limits for plotting
        if time_limits == [-1, -1]:
            self.__time_limits = [0, self.__period/2]
        else:
            self.__time_limits = time_limits
            
        # Time axis data for plot
        self.__time_axis = np.arange(0, self.__time_limits[1] + self.__timestep, self.__timestep)
        
        # Time array and data, sliding window based on gating delay
        self.__time_array = np.zeros((self.__number_of_chips, self.__number_of_bins), dtype=float)
        self.__time_data = np.zeros_like(self.__time_array, dtype=np.intc)
        
        # Time data that has been adjusted ot full axis bounds
        self.__transformed_time_data = np.zeros((self.__number_of_chips, len(self.__time_axis)), dtype= np.intc)
        
        # Data structure for FIFO data storage
        self.__data = np.empty((self.__number_of_chips, self.__number_of_frames, self.__patt_per_frame, self.__number_of_bins), dtype=np.intc)
        
        # Data structures for average data storage
        self.__capture_data = np.empty((self.__number_of_chips, self.__number_of_bins), dtype=np.intc)
        self.__capture_std = np.empty_like(self.__capture_data, dtype=float)
        self.__capture_std_data = np.empty((self.__number_of_chips, self.__number_of_bins, self.__number_of_frames * self.__patt_per_frame - 1))
        self.__frame_totals = np.empty((self.__number_of_chips, self.__number_of_frames, self.__patt_per_frame), dtype=np.intc)
        self.__frame_total_mean = np.empty((self.__number_of_chips), dtype=np.int)
        self.__frame_total_std = np.empty((self.__number_of_chips), dtype=np.int)
        
        # Data structures for persistence
        self.__persistent_data = np.zeros((self.__number_of_chips, self.__persistence_depth, len(self.__time_axis)), dtype=np.intc)
        self.__persistent_averages = np.zeros((self.__number_of_chips, self.__persistence_depth))
        self.__persistence_mask = [[False] * self.__persistence_depth] * self.__number_of_chips
        self.__persistent_plot_data =  np.empty((self.__number_of_chips, len(self.__time_axis)), dtype=np.intc)
        self.__persistent_data_to_plot = np.empty((self.__number_of_chips, len(self.__time_axis)), dtype=float)
        
        # Accumulated data
        self.__accumulated_data = np.zeros((self.__number_of_chips, len(self.__time_axis)), dtype=np.intc)
        
        # Regression
        self.__reg = LinearRegression()
        
        # Total counts
        self.__total_counts_data = np.empty((self.__number_of_chips, self.__number_of_frames, self.__patt_per_frame), dtype=np.intc)
        self.__average_total_counts = np.empty((self.__number_of_chips), dtype=float)
        
        # Mean values
        self.__mean_time = np.zeros((self.__number_of_chips), dtype=float)
        self.__mean_persistent = np.zeros((self.__number_of_chips), dtype=float)
        self.__mean_accumulated = np.zeros((self.__number_of_chips), dtype=float)
        
        # Gating delays
        self.__gate_delay = np.zeros((self.__number_of_chips), dtype=float)
        self.__delay_line_word = np.zeros((self.__number_of_chips), dtype=np.intc)
        self.__coarse = ['0000'] * self.__number_of_chips
        self.__fine = ['000'] * self.__number_of_chips

        # Figure handles
        self.__plot_figure = None
        self.__plot_line = [None] * self.__number_of_chips
        self.__plot_axes = [None] * self.__number_of_chips
        self.__subplot_spawned = [False] * self.__number_of_chips
        self.__plot_peak_spawned = [False] * self.__number_of_chips
        self.__peak_point = [None] * self.__number_of_chips
        self.__plot_peak_text = [None] * self.__number_of_chips
        self.__plot_info_text_spawned = [False] * self.__number_of_chips
        self.__plot_info_text = [None] * self.__number_of_chips
        self.__plot_mean_text = [None] * self.__number_of_chips
        
        # Accessory line handles
        self.__vcsel_line_spawned = [False] * self.__number_of_chips
        self.__vcsel_line = [None] * self.__number_of_chips
        self.__gating_line_spawned = [False] * self.__number_of_chips
        self.__gating_line = [None] * self.__number_of_chips
        self.__mean_line_spawned = [False] * self.__number_of_chips
        self.__mean_line = [None] * self.__number_of_chips
        
        # Shape of subplots
        self.__subplot_rows = int(np.ceil(np.sqrt(self.__number_of_chips)))
        self.__subplot_cols = int(np.ceil(np.sqrt(self.__number_of_chips)))
        if (self.__subplot_rows - 1) * self.__subplot_cols >= self.__number_of_chips:
            self.__subplot_rows = self.__subplot_rows - 1
        
        # Pointer for persistent data circular buffer
        self.__cb_pointer = np.zeros((self.__number_of_chips), dtype=np.intc)

        
    # ===========================================================
    # Interpret the fifo data. Separate information by frame, pattern, and bin
    # ===========================================================
    def __interpret(self, fifo_data):
        
        # Reset data to zero
        self.__data.fill(0)
        
        # Loop through the FIFO data frame-by-frame
        for frame in range(self.__number_of_frames):
            
            # Index where frame data starts and ends within the FIFO data
            frame_start_index = self.__transfer_size * frame * self.__patt_per_frame
            
            # Loop through the frame data pattern-by-pattern
            for pattern in range(self.__patt_per_frame):
                
                # Index where pattern data starts and ends within the frame data
                pattern_start_index = frame_start_index + self.__transfer_size * pattern
                pattern_end_index = pattern_start_index + self.__transfer_size
                
                # For calculating total counts
                counts_in_pattern = [0] * self.__number_of_chips
                
                # Loop through the pattern data bin-by-bin
                for bin_number in range(self.__number_of_bins):
                    
                    # Index where bin data starts and ends within the pattern
                    bin_data_end_index = pattern_end_index - bin_number * self.__bin_size
                    bin_data_start_index = bin_data_end_index - self.__bin_size
                    
                    # Loop through chips, data packet by data packet
                    for chip in range(self.__number_of_chips):
                    
                        # Populate the data structure with the value
                        # TODO: Check that __data variable has been extended
                        self.__data[chip][frame][pattern][bin_number] = int(fifo_data[chip][bin_data_start_index:bin_data_end_index], base = 2)
                        
                        # Add to the counts in pattern, skipping the garbage collection bin
                        if bin_number == 0:
                            pass
                        else:
                            counts_in_pattern[chip] = counts_in_pattern[chip] + self.__data[chip][frame][pattern][bin_number]
                        
                    # Store the counts in pattern
                    # TODO: Check that total_counts_data has been extended
                    self.__total_counts_data[chip][frame][pattern] = counts_in_pattern[chip]

    
    # ===========================================================
    # Consider only the last pattern
    # ===========================================================
    def __last_pattern(self):
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Reset averaged_data to zero
            # TODO: Check that capture_data and data were extended
            self.__capture_data[chip] = self.__data[chip][self.__number_of_frames-1][self.__patt_per_frame-1]
            
            # Zero out the zeroeth bin
            self.__capture_data[chip][0] = 0
            
            # Find the total counts
            # TODO: Check that average_total_counts was extended
            self.__average_total_counts[chip] = np.sum(self.__capture_data[chip])
    
    
    # ===========================================================
    # Average the value in each bin by summing values in each bin and dividing by the number of histograms
    # ===========================================================
    def __average(self):
        
        # If large histograms are being cut, averaging must start by finding pattern totals and valid patterns
        if self.cut_large_patterns:
            self.__calculate_pattern_totals()   
            self.__find_valid_patterns()
        
        # Reset averaged_data to zero
        self.__capture_data.fill(0)

        # This variable stores the total number of counts in all of the bins
        total_counts = [0] * self.__number_of_chips
        
        # Array to hold data values for averaging
        data_for_averaging = np.empty((self.__number_of_chips, self.__number_of_frames * self.__patt_per_frame - 1), dtype=np.intc)
        
        # Loop through FIFO data, chip-by-chip
        for chip in range(self.__number_of_chips):
        
            # Loop through each bin in the averaged_data vector
            for bin_number in range(self.__number_of_bins):
                
                # Flat index
                flat_index = 0
                
                # Loop through the data structure frame-by-frame
                for frame in range(self.__number_of_frames):
                    
                    # Loop through the frame data pattern-by-pattern
                    for pattern in range(self.__patt_per_frame):
                        
                        # The first frame and pattern is always 0s, so ignore it
                        # Otherwise, add the counts in this pattern to the overall count for this bin
                        # Increment the total counts by this number as well
                        if (frame == 0) and (pattern == 0):
                            pass
                        else:
                            data_for_averaging[chip][flat_index] = self.__data[chip][frame][pattern][bin_number]
                            total_counts[chip] = total_counts[chip] + self.__data[chip][frame][pattern][bin_number]
                                    
                            # Increment the flat index
                            flat_index = flat_index + 1
                
                # Store the averages
#                if sum(data_for_averaging[chip]) > 30000:
#                    print "Data for averaging"
#                    print data_for_averaging
#                    print "Mask"
#                    print self.__valid_patterns[chip].flatten()[1:self.__number_of_frames * self.__patt_per_frame]
#                    print "After mask:"
#                    print np.mean(data_for_averaging[chip][self.__valid_patterns[chip].flatten()[1:self.__number_of_frames * self.__patt_per_frame]])
                if self.cut_large_patterns:
                    self.__capture_data[chip][bin_number] = int( round( np.mean(data_for_averaging[chip][self.__valid_patterns[chip].flatten()[1:self.__number_of_frames * self.__patt_per_frame]]) , 0) )
                else:
                    self.__capture_data[chip][bin_number] = int( round( np.mean(data_for_averaging[chip]), 0))
                
                # Remove the zeroeth bin (this is a garbage collection bin)
                self.__capture_data[chip][0] = 0
                
                # Store the standard deviations
                # TODO: Standard deviation calculation needs to be updated to include only data from patterns that were not cut
                self.__capture_std_data[chip][bin_number] = data_for_averaging[chip]
                self.__capture_std[chip][bin_number] = np.std(data_for_averaging[chip])
            
            # Calculate the average total hits
            # TODO: Check that self.__average_total_counts has been extended
            self.__average_total_counts[chip] = int(round(total_counts[chip] / (self.__number_of_frames*self.__patt_per_frame-1), 0))
        

    # ===========================================================
    # Calculate the total counts in each pattern
    # ===========================================================
    def __calculate_pattern_totals(self):
        
        # Reset averaged_data to zero
        # TODO: Check that self.__frame_totals has been extended
        self.__frame_totals.fill(0)
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Loop through the FIFO data frame-by-frame
            for frame in range(self.__number_of_frames):
                
                # Loop through the frame data pattern-by-pattern
                for pattern in range(self.__patt_per_frame):
                    
                    # Find the sum of all the bins in the pattern
                    self.__frame_totals[chip][frame][pattern] = np.sum(self.__data[chip][frame][pattern])

            # Find average and standard deviation of frame totals if cutting large histograms
            if self.cut_large_patterns:
                self.__frame_total_mean[chip] = int(np.mean(self.__frame_totals[chip]))
                self.__frame_total_std[chip] = int(np.std(self.__frame_totals[chip]))
                
 
    # ===========================================================
    # Create a list of valid patterns (used when cutting large histograms)
    # ===========================================================
    def __find_valid_patterns(self):
        
        # Data structure for recording whether a pattern is valid
        self.__valid_patterns = np.empty((self.__number_of_chips, self.__number_of_frames, self.__patt_per_frame), dtype=np.bool)
        
        # Loop through pattern totals, chip by chip
        for chip in range(self.__number_of_chips):
            
            # Find the exclusion value for this chip
            exclude = self.__frame_total_mean[chip] + self.__frame_total_std[chip] * self.__number_stds_to_include
            
            # Loop through chip pattern totals, frame by frame
            for frame in range(self.__number_of_frames):
                
                # Loop through frame pattern totals, pattern by pattern
                for pattern in range(self.__patt_per_frame):
                    
                    # Pattern is valid if the number of counts in the pattern is below the exclusion boundary
                    if self.__frame_totals[chip][frame][pattern] < exclude:
                        self.__valid_patterns[chip][frame][pattern] = True
                    else:
                        self.__valid_patterns[chip][frame][pattern] = False
                        
    
    # ===========================================================
    # Print the total counts in each pattern
    # ===========================================================
    def __print_pattern_totals(self):   
        
        # Calculate pattern totals - if averaging and cut_large_patterns are enabled, this will already be done
        if not (self.averaging & self.cut_large_patterns):
            self.__calculate_pattern_totals()
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Print the chip header
            print("Chip {}, ".format(chip)),
    
            # Loop through the FIFO data, frame-by-frame
            for frame in range(self.__number_of_frames):
                
                # Print the frame header
                print("Frame {}:".format(frame)),
                
                # Loop through the frame data, pattern-by-pattern
                for pattern in range(self.__patt_per_frame):
                    
                    # Print the total for each pattern
                    print(str(self.__frame_totals[chip][frame][pattern])),
                    print(","),
                
                # End the line
                print("")
    
    
    # ===========================================================
    # Place the data in time based on the time gating delay
    # ===========================================================
    def __place_data_in_time(self):
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
        
            # Create the time array
            # TODO: Allow for unique gating delays for each chip
            self.__time_array[chip] = np.arange(self.__gate_delay[chip], self.__gate_delay[chip] + self.__tdc_resolution * self.__number_of_bins, self.__tdc_resolution)
            
            # Set the time data equal to the capture data
            self.__time_data[chip] = self.__capture_data[chip]
            
            # Remove the 0th bin
            self.__time_data[chip][0] = 0
            
        # Adjust time data to fit axis
        self.__transform_time_data()
        

    # ===========================================================
    # Transform the time data so that it fits the full time axis
    # ===========================================================
    def __transform_time_data(self):
        
        # Reset transformed time data to 0
        self.__transformed_time_data.fill(0)
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
      
            # Loop through the averaged data, bin-by-bin
            for bin_number in range(1, self.__number_of_bins):
                
                # Find the time index based on the gating delay
                time_index = int( round((self.__gate_delay[chip] + bin_number * self.__tdc_resolution) / self.__timestep, 0))
                
                # Place the counts for the bin into the time_data array
                if (time_index < len(self.__time_axis)):
                    self.__transformed_time_data[chip][time_index] = self.__time_data[chip][bin_number]
                
    
    # ===========================================================
    # Interpolate the time data
    # ===========================================================
    def __interpolate_transformed_time_data(self):
        pass
                
    
    # ===========================================================
    # Find the mean of the plotted data
    # ===========================================================
    def __calculate_mean_time_data(self):
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
        
            # Need to verify that transformed time data does not sum to zero, or this will lead to division by zero
            if bool(np.sum(self.__transformed_time_data[chip])):
                
                # Average the data
                # TODO: Check that self.__mean_time has been extended
                self.__mean_time[chip] = np.average(self.__time_axis, weights=self.__transformed_time_data[chip])
                
            else:
                
                # No data means the mean is 0
                self.__mean_time[chip] = 0.0
        
    
    # ===========================================================
    # Find the mean of the accumulated data
    # ===========================================================
    def __calculate_mean_accumulated_data(self, chip):
        
        # Verify that accumulated data is not 0
        if bool(np.sum(self.__accumulated_data[chip])):
            
            # Calculate the mean
            self.__mean_accumulated[chip] = np.average(self.__time_axis, weights=self.__accumulated_data[chip])
            
        else:
            
            # Set persistent mean to 0
            self.__mean_accumulated[chip] = 0
            
    
    # ===========================================================
    # Find the mean of the persistent data
    # ===========================================================
    def __calculate_mean_persistent_data(self):
        
        # Reset the mean data
        self.__persistent_plot_data.fill(0)
        
        # Loop through the persistent data, dataset-by-dataset
        for dataset in range(self.__persistence_depth):
            
            # Add to accumulated data
            self.__persistent_plot_data = np.maximum(self.__persistent_plot_data, self.__persistent_data[dataset])
            
        # Add the latest current data to the accumulated data
        self.__persistent_plot_data = np.maximum(self.__persistent_plot_data, self.__transformed_time_data)
        
        # Verify that accumulated data is not 0
        if bool(np.sum(self.__persistent_plot_data)):
            
            # Calculate the mean
            self.__mean_persistent = np.average(self.__time_axis, weights=self.__persistent_plot_data)
            
        else:
            
            # Set persistent mean to 0
            self.__mean_persistent = 0
        
    
    # ===========================================================
    # Store the current data
    # ===========================================================
    def __store_data(self):
        
        for chip in range(self.__number_of_chips):
        
            # Store the data
            self.__persistent_data[chip][self.__cb_pointer[chip]] = self.__transformed_time_data[chip]
            
            # Store the mean
            self.__persistent_averages[chip][self.__cb_pointer[chip]] = self.__mean_time[chip]
            
            # Indicate the persistence slot is used
            self.__persistence_mask[chip][self.__cb_pointer[chip]] = True
            
            # Increment the pointer
            if (self.__cb_pointer[chip] < self.__persistence_depth - 1):
                self.__cb_pointer[chip] = self.__cb_pointer[chip] + 1
            else:
                self.__cb_pointer[chip] = 0

        
    # ===========================================================        
    # Plot raw bin data fast
    # ===========================================================
    def __plot_raw_fast(self):
        
        if not self.__plot_figure_spawned:
            
            # Close existing figures
            plt.close('all')
        
            # Figure handle
            self.__plot_figure, axes_structure = plt.subplots(self.__subplot_rows, self.__subplot_cols, sharex=True, sharey=True)
            
            # Maximize figure
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            
            # Label graphs
            for ax in range(self.__number_of_chips):
                
                self.__plot_axes[ax] = axes_structure.flat[ax]
                
                # Subplot title
                self.__plot_axes[ax].set_title('Chip ' + str(ax))
                
                # Subplot axis labels
                self.__plot_axes[ax].set(xlabel='Bin Number', ylabel='Counts')
                
                # Show units only on outer plots
                self.__plot_axes[ax].label_outer()
                
                # Set x-axis range
                self.__plot_axes[ax].set_xlim((0, 150))
                
            # Indicate plot has been spawned
            self.__plot_figure_spawned = True
            
        # Plot title with updated capture number
        self.__plot_figure.suptitle("Capture {}".format(self.__current_capture))

        # Loop through chips
        for chip in range(self.__number_of_chips):
        
            # Spawn the figure
            if not self.__subplot_spawned[chip]:
                
                # Plot the data
                self.__plot_line[chip] = self.__plot_axes[chip].plot(range(1, self.__number_of_bins), self.__capture_data[chip][1:self.__number_of_bins], color='#00008B', marker='.')[0]
                             
                # Indicate the figure has been spawned
                self.__subplot_spawned[chip] = True
                
            else:
                
                # Update the line
                self.__plot_line[chip].set_ydata(self.__capture_data[chip][1:self.__number_of_bins])
                
        # Update y-axis range of all figures
        self.__plot_axes[0].set_ylim([0, np.round(np.amax(self.__capture_data) * 1.1 + 1)])
                
        # Redraw the plot figure
        self.__plot_figure.canvas.draw()
        self.__plot_figure.canvas.flush_events() 
            
        # Show
        plt.show()
        plt.pause(self.__fast_mode_time)
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1


    # ===========================================================
    # Create a plot of the data wihtout time information
    # ===========================================================
    def __plot_raw(self):
        
        if not self.__plot_figure_spawned:
        
            # Close all figures
            plt.close('all')
            
            # Figure handle
            self.__plot_figure, axes_structure = plt.subplots(self.__subplot_rows, self.__subplot_cols, sharex=True, sharey=True)
            
            # Maximize figure
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            
            # Label graphs
            for ax in range(self.__number_of_chips):
                
                self.__plot_axes[ax] = axes_structure.flat[ax]
                
                # Subplot title
                self.__plot_axes[ax].set_title('Chip ' + str(ax))
                
                # Subplot axis labels
                self.__plot_axes[ax].set(xlabel='Bin Number')
                
                # Label y-axis depending on logarithmic plotting
                if self.plot_logarithmic:
                    self.__plot_axes[ax].set(ylabel="log10(Counts)")
                else:
                    self.__plot_axes[ax].set(ylabel="Counts")
                
                # Show units only on outer plots
                self.__plot_axes[ax].label_outer()
                
                # Set x-axis range
                self.__plot_axes[ax].set_xlim([0, 150])
                
            # Figure y-limits if fixed
            if self.fix_y_max:
                
                # For logarithmic plotting with fixed y-max
                if self.plot_logarithmic:
                    self.__plot_axes[0].set_ylim([0, np.log10(self.__ymax)])
                    
                # For normal plotting with fixed y-max
                else:
                    self.__plot_axes[0].set_ylim([0, self.__ymax])
                                    
            # Indicate plot has been spawned
            self.__plot_figure_spawned = True
            
        # Plot title with updated capture number
        self.__plot_figure.suptitle("Capture {}".format(self.__current_capture))
        
        # Update y-axis range of all figures
        if self.fix_y_max:
            ymin, ymax = 0, self.__ymax 
        else:
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.round(np.amax(self.__capture_data) * 1.1 + 1)])
        
        # Adjust for log setting
        if self.plot_logarithmic and (ymax > 1.0):
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.log10(ymax)])
            
        # If plotting logarithmic, change type
        if self.plot_logarithmic:
            
            # Change datatype
            self.__capture_data = self.__capture_data.astype(float)
            
            # Calculate log
            self.__capture_data = np.log10(self.__capture_data, where=self.__capture_data > 0, out=self.__capture_data)
                        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Spawn the figure
            if not self.__subplot_spawned[chip]:
                
                # Plot the data
                self.__plot_line[chip] = self.__plot_axes[chip].plot(range(150), self.__capture_data[chip], color='#00008B', marker='.')[0]
                             
                # Indicate the figure has been spawned
                self.__subplot_spawned[chip] = True
                
            else:
                
                # Update the line
                self.__plot_line[chip].set_ydata(self.__capture_data[chip])

            # Find the peak
            if self.show_peak:
                
                # Find the peak index
                peak_index = np.argmax(self.__capture_data[chip])
                
                # Store peak value
                peak_value = self.__capture_data[chip][peak_index]
                
                # Place text
                if not self.__plot_peak_spawned[chip]:
                    
                    # Spawn peak point and text
                    self.__peak_point[chip] = self.__plot_axes[chip].plot(peak_index, peak_value, 'ro')[0]
                    self.__plot_peak_text[chip] = self.__plot_axes[chip].text(peak_index + self.__time_limits[1] * .02, peak_value, "Bin " + str(peak_index))
                    self.__plot_peak_spawned[chip] = True
                    
                else:
                    
                    # Replot peak point and text
                    self.__peak_point[chip].set_ydata(peak_value)
                    self.__plot_peak_text[chip].set_position([peak_index + self.__time_limits[1] * .02, peak_value])
                    self.__plot_peak_text[chip].set_text("Bin " + str(peak_index))
            
            # Add figure text to indicate the counts
            if (self.show_plot_info):
                if not self.__plot_info_text_spawned[chip]:
                    self.__plot_info_text[chip] = self.__plot_axes[chip].text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]), fontsize=8, verticalalignment='top')
                    self.__plot_info_text_spawned[chip] = True
                else:
                    self.__plot_info_text[chip].set_text("Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]))
                    self.__plot_info_text[chip].set_position([self.__time_limits[1] * .05, ymax * .95])

        # Assemble the legend and handles
        legend = ['Data']
        handles = [self.__plot_line[0]]
        if self.show_peak:
            legend.append('Peak')
            handles.append(self.__peak_point[0])
        
        # Add a legend to the plot
        if not self.__plot_legend_spawned:
            self.__plot_figure.legend(handles, legend, loc='upper right', fontsize='small')
            self.__plot_legend_spawned = True
        
        # Save the figure
        if self.save_figures:
            self.__plot_figure.savefig(self.__plots_dir + "Capture_{0:0=5d}.png".format(self.__current_capture), dpi=100)
        
        # Show
        plt.show()
        plt.pause(0.1)
        time.sleep(0.1)
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1



    
    # ===========================================================
    # Create a plot of the data
    # ===========================================================
    def __plot(self):
        
        if not self.__plot_figure_spawned:
        
            # Close all figures
            plt.close('all')
            
            # Figure handle
            self.__plot_figure, axes_structure = plt.subplots(self.__subplot_rows, self.__subplot_cols, sharex=True, sharey=True)
            
            # Maximize figure
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            
            # Label graphs
            for ax in range(self.__number_of_chips):
                
                self.__plot_axes[ax] = axes_structure.flat[ax]
                
                # Subplot title
                self.__plot_axes[ax].set_title('Chip ' + str(ax))
                
                # Subplot axis labels
                self.__plot_axes[ax].set(xlabel='Time (ns)')
                
                # Label y-axis depending on logarithmic plotting
                if self.plot_logarithmic:
                    self.__plot_axes[ax].set(ylabel="log10(Counts)")
                else:
                    self.__plot_axes[ax].set(ylabel="Counts")
                
                # Show units only on outer plots
                self.__plot_axes[ax].label_outer()
                
                # Set x-axis range
                self.__plot_axes[ax].set_xlim(self.__time_limits)
                
            # Figure y-limits if fixed
            if self.fix_y_max:
                
                # For logarithmic plotting with fixed y-max
                if self.plot_logarithmic:
                    self.__plot_axes[0].set_ylim([0, np.log10(self.__ymax)])
                    
                # For normal plotting with fixed y-max
                else:
                    self.__plot_axes[0].set_ylim([0, self.__ymax])
                                    
            # Indicate plot has been spawned
            self.__plot_figure_spawned = True
            
        # Plot title with updated capture number
        self.__plot_figure.suptitle("Capture {}".format(self.__current_capture))
        
        # Update y-axis range of all figures
        if self.fix_y_max:
            ymin, ymax = 0, self.__ymax 
        else:
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.round(np.amax(self.__time_data) * 1.1 + 1)])
        
        # Adjust for log setting
        if self.plot_logarithmic and (ymax > 1.0):
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.log10(ymax)])
        
        # Create the VCSEL signal
        if self.__vcsel_setting_set:
            vcsel_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__vcsel_latency, 0.5) - np.heaviside(self.__time_axis - self.__vcsel_latency - self.__vcsel_delays[self.__vcsel_setting], 0.5))
            
        # If showing the mean, calculate means
        if self.show_mean:
            
            # Find the mean
            self.__calculate_mean_time_data()
            
        # If plotting logarithmic, change type
        if self.plot_logarithmic:
            
            # Change datatype
            self.__time_data = self.__time_data.astype(float)
            
            # Calculate log
            self.__time_data = np.log10(self.__time_data, where=self.__time_data > 0, out=self.__time_data)
                        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Spawn the figure
            if not self.__subplot_spawned[chip]:
                
                # Plot the data
                self.__plot_line[chip] = self.__plot_axes[chip].plot(self.__time_array[chip], self.__time_data[chip], color='#00008B', marker='.')[0]
                             
                # Indicate the figure has been spawned
                self.__subplot_spawned[chip] = True
                
            else:
                
                # Update the line
                self.__plot_line[chip].set_ydata(self.__time_data[chip])
                     
            # Perform regression and plot regression line
            if self.plot_logarithmic and self.show_regression:
                
                # Perform regression
                regression_time_array, regression_time_data = self.__regression()
                
                # Plot best-fit line
                plt.plot(regression_time_array, regression_time_data)
                
                # Place regression line properties
                middle_index = int(len(regression_time_array) / 2)
                plt.text(regression_time_array[middle_index] + self.__time_limits[1] * .1, regression_time_data[middle_index] + ymax * .1, "Slope: {} \nR2: {}".format(self.__slope, self.__rsquared), fontsize=8, verticalalignment='top')
                     
            # Show the mean
            if self.show_mean:
                
                # Plot the mean
                if not self.__mean_line_spawned[chip]:
                    
                    # Spawn mean line and text
                    self.__mean_line[chip] = self.__plot_axes[chip].axvline(self.__mean_time[chip], ymin=0, ymax=4096, color='#9370DB')
                    self.__plot_mean_text[chip] = self.__plot_axes[chip].text(self.__mean_time[chip], ymax * 0.95, str(round(self.__mean_time[chip], 2)) + " ns")
                    self.__mean_line_spawned[chip] = True
                    
                else:
                    
                    # Replot mean line and text
                    self.__mean_line[chip].set_ydata(self.__mean_time[chip])
                    self.__plot_mean_text[chip].set_text(str(round(self.__mean_time[chip], 2)) + " ns")
                    self.__plot_mean_text[chip].set_position([self.__mean_time[chip], ymax * 0.95])
                    
            # Find the peak
            if self.show_peak:
                
                # Find the peak index
                peak_index = np.argmax(self.__time_data[chip])
                
                # Store peak value
                peak_value = self.__time_data[chip][peak_index]
                
                # Find the time that the peak occurs
                peak_time = self.__time_array[chip][peak_index]
                
                # Place text
                if not self.__plot_peak_spawned[chip]:
                    
                    # Spawn peak point and text
                    self.__peak_point[chip] = self.__plot_axes[chip].plot(peak_time, peak_value, 'ro')[0]
                    self.__plot_peak_text[chip] = self.__plot_axes[chip].text(peak_time + self.__time_limits[1] * .02, peak_value, "{} ns".format(round(peak_time, 2)))
                    self.__plot_peak_spawned[chip] = True
                    
                else:
                    
                    # Replot peak point and text
                    self.__peak_point[chip].set_ydata(peak_value)
                    self.__plot_peak_text[chip].set_position([peak_time + self.__time_limits[1] * .02, peak_value])
                    self.__plot_peak_text[chip].set_text("{} ns".format(round(peak_time, 2)))
            
            # If the VCSEL setting has been set, plot it
            if self.__vcsel_setting_set:
                
                # Check if VCSEL signal has already been plotted
                if not self.__vcsel_line_spawned[chip]:
                    
                    # Plot the VCSEL signal
                    self.__vcsel_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, vcsel_signal, color = 'g')[0]
                    
                    self.__vcsel_line_spawned[chip] = True
                    
                else:
                    
                    # Replot VCSEL signal
                    self.__vcsel_line[chip].set_ydata(vcsel_signal)
                    
            # Create and plot the gating signal
            gating_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__spad_rise_time, 0.5) - np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__period / 2, 0.5))
            
            # Check if gating signal has already been plotted
            if not self.__gating_line_spawned[chip]:
                
                # Plot the gating signal
                self.__gating_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, gating_signal, color='r')[0]
                self.__gating_line_spawned[chip] = True
                
            else:
                
                # Replot gating signal
                self.__gating_line[chip].set_ydata(gating_signal)
            
            # Add figure text to indicate the counts
            if (self.show_plot_info):
                if not self.__plot_info_text_spawned[chip]:
                    self.__plot_info_text[chip] = self.__plot_axes[chip].text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]), fontsize=8, verticalalignment='top')
                    self.__plot_info_text_spawned[chip] = True
                else:
                    self.__plot_info_text[chip].set_text("Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]))
                    self.__plot_info_text[chip].set_position([self.__time_limits[1] * .05, ymax * .95])

        # Assemble the legend and handles
        legend = ['Data']
        handles = [self.__plot_line[0]]
        if self.plot_logarithmic and self.show_regression:
            legend.append("Reg. Line")
        if self.show_mean:
            legend.append('Mean')
            handles.append(self.__mean_line[0])
        if self.show_peak:
            legend.append('Peak')
            handles.append(self.__peak_point[0])
        legend.append('Gating Window')
        handles.append(self.__gating_line[0])
        if self.__vcsel_setting_set:
            legend.append('VCSEL Pulse')
            handles.append(self.__vcsel_line[0])
        
        # Add a legend to the plot
        if not self.__plot_legend_spawned:
            self.__plot_figure.legend(handles, legend, loc='upper right', fontsize='small')
            self.__plot_legend_spawned = True
        
        # Save the figure
        if self.save_figures:
            self.__plot_figure.savefig(self.__plots_dir + "Capture_{0:0=5d}.png".format(self.__current_capture), dpi=100)
        
        # Show
        plt.show()
        plt.pause(0.1)
        time.sleep(0.1)
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1


    # ===========================================================
    # Called when you want plot data returned with no plot generated
    # ===========================================================
    def __plot_without_plot(self):
        
        # If showing the mean, calculate means
        if self.show_mean:
            
            # Find the mean
            self.__calculate_mean_time_data()
            
        # If plotting logarithmic, change type
        if self.plot_logarithmic:
            
            # Change datatype
            self.__time_data = self.__time_data.astype(float)
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
        
            # Plot logarithmic
            if self.plot_logarithmic:
                
                # Calculate log
                self.__time_data[chip] = np.log10(self.__time_data[chip], where=self.__time_data[chip] > 0, out=self.__time_data[chip])
                     
            # Perform regression and plot regression line
            if self.plot_logarithmic and self.show_regression:
                
                # Perform regression
                regression_time_array, regression_time_data = self.__regression()
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1
        

    # ===========================================================
    # Create a plot of the data
    # NOT SUPPORTED
    # ===========================================================
    def __plot_persistent(self):
        
        # Clear the figure
        plt.clf()
        
        # Figure labels
        plt.title("Capture {} Data".format(self.__current_capture))
        plt.xlabel("Time (ns)")
        
        # Label y-axis depending on logarithmic plotting
        if self.plot_logarithmic:
            plt.ylabel("log10(Counts)")
        else:
            plt.ylabel("Counts")
        
        # Figure x-limits
        plt.xlim(self.__time_limits)
        
        # Figure y-limits
        if self.fix_y_max:
            
            # For logarithmic plotting
            if self.plot_logarithmic:
                plt.ylim([0, np.log10(self.__ymax)])
                
            # For normal plotting
            else:
                plt.ylim([0, self.__ymax])
        
        # Create the persistent data to plot vector
        self.__persistent_data_to_plot.fill(0)
        
        # Calculate the data to be shown
        for i in range(self.__persistence_depth):
            self.__persistent_data_to_plot = np.maximum(self.__persistent_data_to_plot, self.__persistent_data[i])
        
        # Calculate logarithmic data
        if self.plot_logarithmic:
            self.__persistent_data_to_plot = np.log10(self.__persistent_data_to_plot, where= self.__persistent_data_to_plot > 0)
        
        # Plot the persistent data
        plt.plot(self.__time_axis, self.__persistent_data_to_plot, color='#ADD8E6')
        
        # Plot the data
        plt.plot(self.__time_array, self.__time_data, color='#00008B')
                 
        # Get the axis and find the max y limit
        ymin, ymax = plt.gca().get_ylim()
        if (ymax < 1):
            ymax = 1
            plt.ylim([0, 1])
            
        # Plot the persistent mean
        if self.show_mean:
            
            # Calculate the mean of persistent data
            self.__calculate_mean_persistent_data()
        
            # Plot the mean
            plt.axvline(self.__mean_persistent, ymin=ymin, ymax=ymax, color='#9370DB')
            plt.text(self.__mean_persistent, ymax * 0.95, str(round(self.__mean_persistent, 2)) + " ns")
        
        # Find the peak
        if self.show_peak:
            
            # Find the peak index
            peak_index = np.argmax(self.__persistent_data_to_plot)
            
            # Store peak value
            if self.plot_logarithmic:
                peak_value = np.log10(self.__persistent_data_to_plot[peak_index])
            else:
                peak_value = self.__persistent_data_to_plot[peak_index]
            
            # Find the time that the peak occurs
            peak_time = self.__time_axis[peak_index]
            
            # Plot a point
            plt.plot(peak_time, peak_value, 'ro')
            
            # Place some text
            plt.text(peak_time + self.__time_limits[1] * .02, peak_value, "{} ns".format(round(peak_time, 2)))
        
        # Create and plot the gating signal
        gating_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__spad_rise_time, 0.5) - np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__period / 2, 0.5))
        plt.plot(self.__time_axis, gating_signal, color='r')
        
        # Create and plot the VCSEL signal
        if self.__vcsel_setting_set:
            vcsel_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__vcsel_latency, 0.5) - np.heaviside(self.__time_axis - self.__vcsel_latency - self.__vcsel_delays[self.__vcsel_setting], 0.5))
            plt.plot(self.__time_axis, vcsel_signal, color = 'g')
        
        # Add figure text to indicate the counts
        if (self.show_plot_info):
            plt.text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts, self.__subtractor_value, self.__delay_line_word, self.__coarse, self.__fine), fontsize=8, verticalalignment='top')
        else:
            plt.text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}".format(self.__average_total_counts, self.__subtractor_value, self.__delay_line_word), fontsize=8, verticalalignment='top')
           
        # Assemble the legend
        legend = ['Old Data', 'New Data']
        if self.show_mean:
            legend.append('Mean')
        if self.show_peak:
            legend.append('Peak')
        legend.append('Gating Window')
        if self.__vcsel_setting_set:
            legend.append('VCSEL Pulse')
        
        # Add a legend to the plot
        plt.legend(legend, loc='upper right', fontsize='small')
        
        # Save the figure
        if self.save_figures:
            plt.savefig(self.__plots_dir + "Image_{0:0=5d}.png".format(self.__current_capture), dpi=100)
        
        # Show
        plt.show()
        plt.pause(0.5)
        time.sleep(0.5)
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1
        
        # Store this data in the circular buffer
        self.__store_data()
        
        
    # ===========================================================
    # Create a plot of the data, accumulating results every frame
    # ===========================================================
    def __plot_accumulated(self):
        
        if not self.__plot_figure_spawned:
        
            # Close all figures
            plt.close('all')
            
            # Figure handle
            self.__plot_figure, axes_structure = plt.subplots(self.__subplot_rows, self.__subplot_cols, sharex=True, sharey=True)
            
            # Maximize figure
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            
            # Label graphs
            for ax in range(self.__number_of_chips):
                
                self.__plot_axes[ax] = axes_structure.flat[ax]
                
                # Subplot title
                self.__plot_axes[ax].set_title('Chip ' + str(ax))
                
                # Subplot axis labels
                self.__plot_axes[ax].set(xlabel='Time (ns)')
                
                # Label y-axis depending on logarithmic plotting
                if self.plot_logarithmic:
                    self.__plot_axes[ax].set(ylabel="log10(Counts)")
                else:
                    self.__plot_axes[ax].set(ylabel="Counts")
                
                # Show units only on outer plots
                self.__plot_axes[ax].label_outer()
                
                # Set x-axis range
                self.__plot_axes[ax].set_xlim(self.__time_limits)
                
            # Figure y-limits if fixed
            if self.fix_y_max:
                
                # For logarithmic plotting with fixed y-max
                if self.plot_logarithmic:
                    self.__plot_axes[0].set_ylim([0, np.log10(self.__ymax)])
                    
                # For normal plotting with fixed y-max
                else:
                    self.__plot_axes[0].set_ylim([0, self.__ymax])
                                    
            # Indicate plot has been spawned
            self.__plot_figure_spawned = True
            
        # Accumulate data
        self.__accumulated_data = np.add(self.__accumulated_data, self.__transformed_time_data)
            
        # Plot title with updated capture number
        self.__plot_figure.suptitle("Capture {}".format(self.__current_capture))
        
        # Update y-axis range of all figures
        if self.fix_y_max:
            ymin, ymax = 0, self.__ymax 
        else:
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.round(np.amax(self.__accumulated_data) * 1.1 + 1)])
        
        # Adjust for log setting
        if self.plot_logarithmic and (ymax > 1.0):
            ymin, ymax = self.__plot_axes[0].set_ylim([0, np.log10(ymax)])
        
        # Create the VCSEL signal
        if self.__vcsel_setting_set:
            vcsel_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__vcsel_latency, 0.5) - np.heaviside(self.__time_axis - self.__vcsel_latency - self.__vcsel_delays[self.__vcsel_setting], 0.5))
            
        # If showing the mean, calculate means
        if self.show_mean:
            
            # Find the mean
            self.__calculate_mean_time_data()
            
        # If plotting logarithmic, change type
        if self.plot_logarithmic:
            
            # Change datatype
            self.__time_data = self.__time_data.astype(float)
            
            # Calculate log
            self.__time_data = np.log10(self.__time_data, where=self.__time_data > 0, out=self.__time_data)
            
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Spawn the figure
            if not self.__subplot_spawned[chip]:
                
                # Plot the data
                if self.plot_logarithmic:
                    self.__plot_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, np.log10(self.__accumulated_data[chip].astype(float), where= self.__accumulated_data[chip] > 0, out=np.zeros_like(self.__accumulated_data[chip], dtype=float)), color='#00008B')[0]
                else:
                    self.__plot_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, self.__accumulated_data[chip], color='#00008B')[0]
                             
                # Indicate the figure has been spawned
                self.__subplot_spawned[chip] = True
                
            else:
                
                # Update the line
                if self.plot_logarithmic:
                    self.__plot_line[chip].set_ydata(np.log10(self.__accumulated_data[chip].astype(float), where= self.__accumulated_data[chip] > 0, out=np.zeros_like(self.__accumulated_data[chip], dtype=float)))
                else:
                    self.__plot_line[chip].set_ydata(self.__accumulated_data[chip])
                     
            # Perform regression and plot regression line
            if self.plot_logarithmic and self.show_regression:
                
                # Perform regression
                regression_time_array, regression_time_data = self.__regression()
                
                # Plot best-fit line
                plt.plot(regression_time_array, regression_time_data)
                
                # Place regression line properties
                middle_index = int(len(regression_time_array) / 2)
                plt.text(regression_time_array[middle_index] + self.__time_limits[1] * .1, regression_time_data[middle_index] + ymax * .1, "Slope: {} \nR2: {}".format(self.__slope, self.__rsquared), fontsize=8, verticalalignment='top')
                     
            # Show the mean
            if self.show_mean:
                
                # Plot the mean
                if not self.__mean_line_spawned[chip]:
                    
                    # Spawn mean line and text
                    self.__mean_line[chip] = self.__plot_axes[chip].axvline(self.__mean_time[chip], ymin=0, ymax=4096, color='#9370DB')
                    self.__plot_mean_text[chip] = self.__plot_axes[chip].text(self.__mean_time[chip], ymax * 0.95, str(round(self.__mean_time[chip], 2)) + " ns")
                    self.__mean_line_spawned[chip] = True
                    
                else:
                    
                    # Replot mean line and text
                    self.__mean_line[chip].set_ydata(self.__mean_time[chip])
                    self.__plot_mean_text[chip].set_text(str(round(self.__mean_time[chip], 2)) + " ns")
                    self.__plot_mean_text[chip].set_position([self.__mean_time[chip], ymax * 0.95])
                    
            # Find the peak
            if self.show_peak:
                
                # Find the peak index
                peak_index = np.argmax(self.__accumulated_data[chip])
                
                # Store peak value
                if self.plot_logarithmic:
                    peak_value = np.log10(self.__accumulated_data[chip][peak_index])
                else:
                    peak_value = self.__accumulated_data[chip][peak_index]
                
                # Find the time that the peak occurs
                peak_time = self.__time_axis[peak_index]
                
                # Place text
                if not self.__plot_peak_spawned[chip]:
                    
                    # Spawn peak point and text
                    self.__peak_point[chip] = self.__plot_axes[chip].plot(peak_time, peak_value, 'ro')[0]
                    self.__plot_peak_text[chip] = self.__plot_axes[chip].text(peak_time + self.__time_limits[1] * .02, peak_value, "{} ns".format(round(peak_time, 2)))
                    self.__plot_peak_spawned[chip] = True
                    
                else:
                    
                    # Replot peak point and text
                    self.__peak_point[chip].set_ydata(peak_value)
                    self.__plot_peak_text[chip].set_position([peak_time + self.__time_limits[1] * .02, peak_value])
                    self.__plot_peak_text[chip].set_text("{} ns".format(round(peak_time, 2)))
            
            # If the VCSEL setting has been set, plot it
            if self.__vcsel_setting_set:
                
                # Check if VCSEL signal has already been plotted
                if not self.__vcsel_line_spawned[chip]:
                    
                    # Plot the VCSEL signal
                    self.__vcsel_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, vcsel_signal, color = 'g')[0]
                    
                    self.__vcsel_line_spawned[chip] = True
                    
                else:
                    
                    # Replot VCSEL signal
                    self.__vcsel_line[chip].set_ydata(vcsel_signal)
                    
            # Create and plot the gating signal
            gating_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__spad_rise_time, 0.5) - np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__period / 2, 0.5))
            
            # Check if gating signal has already been plotted
            if not self.__gating_line_spawned[chip]:
                
                # Plot the gating signal
                self.__gating_line[chip] = self.__plot_axes[chip].plot(self.__time_axis, gating_signal, color='r')[0]
                self.__gating_line_spawned[chip] = True
                
            else:
                
                # Replot gating signal
                self.__gating_line[chip].set_ydata(gating_signal)
            
            # Add figure text to indicate the counts
            if (self.show_plot_info):
                if not self.__plot_info_text_spawned[chip]:
                    self.__plot_info_text[chip] = self.__plot_axes[chip].text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]), fontsize=8, verticalalignment='top')
                    self.__plot_info_text_spawned[chip] = True
                else:
                    self.__plot_info_text[chip].set_text("Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]))
                    self.__plot_info_text[chip].set_position([self.__time_limits[1] * .05, ymax * .95])

        # Assemble the legend and handles
        legend = ['Data']
        handles = [self.__plot_line[0]]
        if self.plot_logarithmic and self.show_regression:
            legend.append("Reg. Line")
        if self.show_mean:
            legend.append('Mean')
            handles.append(self.__mean_line[0])
        if self.show_peak:
            legend.append('Peak')
            handles.append(self.__peak_point[0])
        legend.append('Gating Window')
        handles.append(self.__gating_line[0])
        if self.__vcsel_setting_set:
            legend.append('VCSEL Pulse')
            handles.append(self.__vcsel_line[0])
        
        # Add a legend to the plot
        if not self.__plot_legend_spawned:
            self.__plot_figure.legend(handles, legend, loc='upper right', fontsize='small')
            self.__plot_legend_spawned = True
        
        # Save the figure
        if self.save_figures:
            self.__plot_figure.savefig(self.__plots_dir + "Capture_{0:0=5d}.png".format(self.__current_capture), dpi=100)
        
        # Show
        plt.show()
        plt.pause(0.1)
        time.sleep(0.1)
        
        # Increment the capture count
        self.__current_capture = self.__current_capture + 1
        
    
#    # ===========================================================
#    # Create a plot of the data
#    # ===========================================================
#    def __plot_accumulated(self):
#        
#        # Clear the figure
#        plt.close('all')
#        
#        # Loop through chips
#        for chip in range(self.__number_of_chips):
#            
#            # Create figure
#            plt.figure()
#        
#            # Figure labels
#            plt.title("Chip {}, Capture {} Data".format(chip, self.__current_capture))
#            plt.xlabel("Time (ns)")
#            
#            # Label y-axis depending on logarithmic plotting
#            if self.plot_logarithmic:
#                plt.ylabel("log10(Counts)")
#            else:
#                plt.ylabel("Counts")
#            
#            # Figure x-limits
#            plt.xlim(self.__time_limits)
#            
#            # Figure y-limits
#            if self.fix_y_max:
#                
#                # For logarithmic plotting
#                if self.plot_logarithmic:
#                    plt.ylim([0, np.log10(self.__ymax)])
#                    
#                # For normal plotting
#                else:
#                    plt.ylim([0, self.__ymax])
#            
#            # Accumulate data
#            self.__accumulated_data[chip] = np.add(self.__accumulated_data[chip], self.__transformed_time_data[chip])
#            
#            # Plot the accumulated data
#            if self.plot_logarithmic:
#                plt.plot(self.__time_axis, np.log10(self.__accumulated_data[chip].astype(float), where= self.__accumulated_data[chip] > 0, out=np.zeros_like(self.__accumulated_data[chip], dtype=float)), color='#00008B')
#            else:
#                plt.plot(self.__time_axis, self.__accumulated_data[chip], color='#00008B')
#            
#            # Get the axis and find the max y limit
#            ymin, ymax = plt.gca().get_ylim()
#            if (ymax < 1):
#                ymax = 1
#                plt.ylim([0, 1])
#                
#            # Plot the accumulated mean
#            if self.show_mean:
#                
#                # Calculate the mean of accumulated data
#                self.__calculate_mean_accumulated_data(chip)
#            
#                # Plot the mean
#                plt.axvline(self.__mean_accumulated[chip], ymin=ymin, ymax=ymax, color='#9370DB')
#                plt.text(self.__mean_accumulated[chip], ymax * 0.95, str(round(self.__mean_accumulated[chip], 2)) + " ns")
#    
#            # Find the peak
#            if self.show_peak:
#                
#                # Find the peak index
#                peak_index = np.argmax(self.__accumulated_data[chip])
#                
#                # Store peak value
#                if self.plot_logarithmic:
#                    peak_value = np.log10(self.__accumulated_data[chip][peak_index])
#                else:
#                    peak_value = self.__accumulated_data[chip][peak_index]
#                
#                # Find the time that the peak occurs
#                peak_time = self.__time_axis[peak_index]
#                
#                # Plot a point
#                plt.plot(peak_time, peak_value, 'ro')
#                
#                # Place some text
#                plt.text(peak_time + self.__time_limits[1] * .02, peak_value, "{} ns".format(round(peak_time, 2)))
#    
#            # Create and plot the gating signal
#            gating_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__spad_rise_time, 0.5) - np.heaviside(self.__time_axis - self.__gate_delay[chip] - self.__period / 2, 0.5))
#            plt.plot(self.__time_axis, gating_signal, color='r')
#            
#            # Create and plot the VCSEL signal
#            if self.__vcsel_setting_set:
#                vcsel_signal = 0.75 * ymax * (np.heaviside(self.__time_axis - self.__vcsel_latency, 0.5) - np.heaviside(self.__time_axis - self.__vcsel_latency - self.__vcsel_delays[self.__vcsel_setting], 0.5))
#                plt.plot(self.__time_axis, vcsel_signal, color = 'g')
#            
#            # Add figure text to indicate the counts
#            if (self.show_plot_info):
#                plt.text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}\nCoarse: {}\nFine: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip], self.__coarse[chip], self.__fine[chip]), fontsize=8, verticalalignment='top')
#            else:
#                plt.text(self.__time_limits[1] * .05, ymax * .95, "Total Counts: {} \nSubtractor: {}\nDelay Word: {}".format(self.__average_total_counts[chip], self.__subtractor_value, self.__delay_line_word[chip]), fontsize=8, verticalalignment='top')
#                  
#            # Assemble the legend
#            legend = ['Data']
#            if self.show_mean:
#                legend.append('Mean')
#            if self.show_peak:
#                legend.append('Peak')
#            legend.append('Gating Window')
#            if self.__vcsel_setting_set:
#                legend.append('VCSEL Pulse')
#            
#            # Add a legend to the plot
#            plt.legend(legend, loc='upper right', fontsize='small')
#            
#            # Save the figure
#            if self.save_figures:
#                plt.savefig(self.__plots_dir + "Chip{0:0=2d}_Image_{1:0=5d}.png".format(chip, self.__current_capture), dpi=100)
#            
#            # Show
#            plt.show(0.5)
#            plt.pause(0.5)
#        
#        # Increment the capture count
#        self.__current_capture = self.__current_capture + 1
        
        
    # ===========================================================
    # Perform linear regression of data - only normal data plotting is supported
    # NOT SUPPORTED
    # ===========================================================
    def __regression(self):
        
        # Find the index where the time gating cuts off the counts
        gate_cutoff_index = int(self.__period / 2 / self.__tdc_resolution)
        
        # Find the index of the maximum value of the time data
        peak_index = np.argmax(self.__time_data[0:gate_cutoff_index])
        
#        # Truncate the data to include only values after the peak
#        truncated_time_array = self.__time_array[peak_index:gate_cutoff_index-2].reshape(-1, 1)
#        truncated_time_data = self.__time_data[peak_index:gate_cutoff_index-2].reshape(-1, 1)
        
        # Truncate the data to include only values halfway past the peak
        halfway_past_peak_index = int((gate_cutoff_index + peak_index) / 2)
        truncated_time_array = self.__time_array[halfway_past_peak_index:gate_cutoff_index-2].reshape(-1, 1)
        truncated_time_data = self.__time_data[halfway_past_peak_index:gate_cutoff_index-2].reshape(-1, 1)
        
        # Perform linear regression
        self.__reg.fit(truncated_time_array, truncated_time_data)
        
        # Store coefficients
        self.__slope = round(self.__reg.coef_[0][0], 4)
        self.__intercept = round(self.__reg.intercept_[0], 4)
        
        # Predict linear regression line
        predicted_time_data = self.__reg.predict(truncated_time_array)
        
        # Compute the r^2 value
        self.__rsquared = round(r2_score(truncated_time_data, predicted_time_data), 4)
        
        # Return the truncated time array and linear regression line for plotting
        return truncated_time_array, predicted_time_data
               
    
    
    # ===========================================================
    # Set coarse and fine values for display on plots
    # ===========================================================
    def set_coarse_fine(self, coarse, fine):
        
        # Check that time_gating_delays has the correct size
        if (len(coarse) != self.__number_of_chips) or (len(fine) != self.__number_of_chips):
            print("Input to set_coarse_fine should have one coarse and one fine value per chip")
            raise Exception
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
            
            # Set the value
            self.__delay_line_word[chip] = (coarse[chip] << 3) + fine[chip]
            self.__coarse[chip] = np.binary_repr(coarse[chip], 4)
            self.__fine[chip] = np.binary_repr(fine[chip], 3)
            
            
    # ===========================================================
    # Set subtractor value
    # ===========================================================
    def set_subtractor_value(self, subtractor_value):
        # Set the value
        self.__subtractor_value = subtractor_value
            
                
    # ===========================================================
    # Set gating delay
    # ===========================================================
    def set_gate_delay(self, time_gating_delays):
        
        # Check that time_gating_delays has the correct size
        if len(time_gating_delays) != self.__number_of_chips:
            print("Input to set_gate_delay should have one delay value per chip")
            raise Exception
        
        # Loop through chips
        for chip in range(self.__number_of_chips):
        
            # Adjust the gate delay to be aligned with current clock cycle
            if (time_gating_delays[chip] + (self.__period / 2) > self.__period):
                self.__gate_delay[chip] = time_gating_delays[chip] - self.__period
            else:
                self.__gate_delay[chip] = time_gating_delays[chip]
    
    
    # ===========================================================
    # Passing the time gate settings for the current frame and the data displays a plot of the current counts
    # ===========================================================
    def update_plot(self, fifo_data):
        
        # Process the data
        self.__interpret(fifo_data)
        
        # Fast mode overrides settings
        if self.fast_mode:
            self.averaging = False
            self.save_figures = False
            self.report_pattern_totals = False
            self.show_plots = True
            self.raw_plotting = True
            self.persistent_plotting = False
            self.accumulated_plotting = False
            self.show_mean = False
            self.show_peak = False
            self.show_plot_info = False
            self.plot_logarithmic = False
            self.show_regression = False
            self.fix_y_max = False
    
        # Average the data
        if self.averaging:
            self.__average()
        else:
            self.__last_pattern()
            
        # Time the data
        if self.raw_plotting is False:
            self.__place_data_in_time()
        
        # Report frame totals
        if self.report_pattern_totals:
            self.__print_pattern_totals()
        
        # Plot the data
        if self.raw_plotting:
            
            # Fast mode raw plotting
            if self.fast_mode:
                
                # Plot quickly
                self.__plot_raw_fast()
            
            else:
            
                # Plot raw bins and counts
                self.__plot_raw()
            
        elif self.persistent_plotting:
            
            # Plot new data over old data
            self.__plot_persistent()
            
        elif self.accumulated_plotting:
            
            # Accumulate data and plot result
            self.__plot_accumulated()
            
        else:
            
            if self.show_plots is True:
                
                # Standard plot
                self.__plot()
                
            else:
                
                # Plot without a plot
                self.__plot_without_plot()
            
            
    # ===========================================================
    # Export processed data with GUI
    # ===========================================================
    def export_processed_data(self):
        
        # # Attempt to get a data file and export directory
        # try:
        #     # Prompt for a CSV file
        #     print("Select a CVS file of FIFO data...")
        #     pathname = fileopenbox()
            
        #     # Attempt to read csv
        #     df = read_csv(pathname)
            
        #     # Prompt for an export directory
        #     print("Select an export directory...")
        #     exports_dir = diropenbox()
            
        #     # If successful, end the loop
        #     print("File read successfully.")
            
        # except IOError:
        #     print("Could not find CSV file or directory.")
        #     exit()
            
        # Process the data - Putting this on hold as it requires a great deal of code
        pass
        
        
    # ===========================================================
    # Turn on or off data averaging
    # ===========================================================
    def set_averaging(self, boolean):
        self.averaging = bool(boolean)
        
        
    # ===========================================================
    # Cut out outlier histograms (with too many counts)
    # ===========================================================
    def set_cut_large_patterns(self, boolean, number_stds=1):
        self.cut_large_patterns = bool(boolean)
        self.__number_stds_to_include = int(number_stds)
        
        
    # ===========================================================
    # Export raw data to specified directory
    # ===========================================================
    def set_export_processed_data(self, boolean, exports_dir):
        
        # Verify that the exports directory exists
        if path.exists(exports_dir):
            self.__exports_dir = exports_dir
            self.export_processed_data = bool(boolean)
        else:
            print("Exports directory does not exist")
            self.export_processed_data = False

    
    # ===========================================================
    # Turn on or off figure saving
    # ===========================================================
    def set_save_figures(self, boolean, plots_dir):

        # Verify that plots directory exists        
        if path.exists(plots_dir):
            self.__plots_dir = plots_dir
            self.save_figures = bool(boolean)
        else:
            print("Plots directory does not exist")
            self.save_figures = False
   
    
    # ===========================================================
    # Turn on or off pattern total printouts
    # ===========================================================
    def set_report_pattern_totals(self, boolean):
        self.report_pattern_totals = bool(boolean)
      
        
    # ===========================================================
    # Turn on or off figure display
    # ===========================================================
    def set_show_plots(self, boolean):
        self.show_plots = bool(boolean)


    # ===========================================================
    # Turn on or off raw plotting
    # ===========================================================
    def set_raw_plotting(self, boolean):
        self.raw_plotting = bool(boolean)

    
    # ===========================================================
    # Turn on or off persistent plotting
    # ===========================================================
    def set_persistent_plotting(self, boolean):
        if boolean is True:
            print("Persistent plotting not supported")
            raise Exception
        
        self.persistent_plotting = bool(boolean)
        
    
    # ===========================================================
    # Turn on or off accumulated plotting
    # ===========================================================
    def set_accumulated_plotting(self, boolean):
        self.accumulated_plotting = bool(boolean)


    # ===========================================================
    # Turn on or off plotted average of persistent data
    # ===========================================================
    def set_show_mean(self, boolean):
        if boolean is True:
            print("Showing mean on plot is not supported")
            raise Exception
        
        self.show_mean = bool(boolean)
     
        
    # ===========================================================
    # Turn on or off plotted average
    # ===========================================================
    def set_show_peak(self, boolean):
        self.show_peak = bool(boolean)
        
        
    # ===========================================================
    # Turn on or off coarse and fine values
    # ===========================================================
    def set_show_plot_info(self, boolean):
        self.show_plot_info = bool(boolean)
        
        
    # ===========================================================
    # Turn on or off logarithmic plotting
    # ===========================================================
    def set_plot_logarithmic(self, boolean):
        self.plot_logarithmic = bool(boolean)
        
        
    # ===========================================================
    # Turn on or off linear regression
    # ===========================================================
    def set_show_regression(self, boolean):
        if boolean is True:
            print("Regression not supported")
            raise Exception
            
        self.show_regression = bool(boolean)
    
    
    # ===========================================================
    # Turn on or off fixed y-max value
    # ===========================================================
    def set_fix_y_max(self, boolean, ymax):
        self.fix_y_max = bool(boolean)
        self.__ymax = float(ymax)
        
        
    # ===========================================================
    # Turn on or off fast mode
    # ===========================================================
    def set_fast_mode(self, boolean, time=0.5):
        self.fast_mode = bool(boolean)
        self.__fast_mode_time = time
    
    # ===========================================================
    # Set the vcsel setting
    # TODO: VCSEL setting should depend on chip that was acting as source during this capture
    # ===========================================================
    def set_vcsel_setting(self, setting):
        
        # Verify that setting is correct
        if (setting < 0) or (setting > 3):
            
            # Default the VCSEL setting to 0
            self.__vcsel_setting = 0
            
            # VCSEL setting becomes unknown
            self.__vcsel_setting_set = False
            
        else:
            
            # Set the VCSEL setting
            self.__vcsel_setting = setting
            
            # VCSEL setting becomes known
            self.__vcsel_setting_set = True
            
            
    # ===========================================================
    # Get the FIFO data
    # ===========================================================
    def get_data(self):
        
        # Return the data
        return self.__data
    
    # ===========================================================
    # Get the plotted data
    # ===========================================================
    def get_plotted_data(self):

        # Return the data from the plot
        if self.raw_plotting:
            return range(self.__number_of_bins), self.__capture_data
        else:
            return self.__time_array, self.__time_data
        
        
    # ===========================================================
    # Get the plotted standard deviations
    # ===========================================================
    def get_plotted_std(self):

        # Return the standard deviation from the plot
        return self.__capture_std
    
    # ===========================================================
    # Get the plotted standard deviation data
    # Used when standard deviations are going to be calculated
    # ===========================================================
    def get_plotted_std_data(self):

        # Return the standard deviation from the plot
        return self.__capture_std_data
        
        
    # ===========================================================
    # Get patterns that fall within the exclusion boundary and are considered valid
    # Only valid if cut_large_patterns is True
    # ===========================================================
    def get_valid_patterns(self):

        # Return the valid patterns if they were found earlier
        if self.cut_large_patterns:
            return self.__valid_patterns
        else:
            print("Valid patterns were not found")
            print("Set cut_large_patterns to True")
            return None
    

    # ===========================================================
    # Get the shape of the time axis
    # ===========================================================
    def get_time_axis_shape(self):
        
        # Return the shape
        return np.shape(self.__time_axis)
    
    
    # ===========================================================
    # Get the shape of the time axis
    # ===========================================================
    def get_time_array_shape(self):
        
        # Return the shape
        return np.shape(self.__time_array)
        
        
    # ===========================================================
    # Get the accumulated data
    # ===========================================================
    def get_accumulated_data(self):
        
        # Check to ensure that data is being accumulated
        if self.accumulated_plotting is False:
            return None, None
        
        # Return the data in the plot
        if self.plot_logarithmic:
            
            # Return the accumulated data on log scale
            return self.__time_axis, np.log10(self.__accumulated_data, where= self.__accumulated_data > 0, out=np.zeros_like(self.__accumulated_data, dtype=float))
        
        else:
            
            # Return the accumulated data
            return self.__time_axis, self.__accumulated_data
        
        
    # ===========================================================
    # Get regression statistics
    # ===========================================================
    def get_regression_statistics(self):
        
        # Return the slope, intercept, and r-squared coefficient
        return self.__slope, self.__intercept, self.__rsquared
    
    
    # ===========================================================
    # Get the average number of counts for this capture
    # ===========================================================
    def get_average_total_counts(self):
    
        # Return the total counts
        return self.__average_total_counts
    
    
    # ===========================================================
    # Get the number of counts in each pattern and frame in this capture
    # ===========================================================
    def get_total_counts_data(self):
    
        # Return the total counts
        return self.__total_counts_data
    
    
    # ===========================================================
    # Close the dataplotter and release all figures
    # ===========================================================
    def close(self):
        plt.close(self.__plot_figure)
    
    
    # ===========================================================
    # Main function for separately exporting data or saving figures
    # ===========================================================
    if __name__ == "__main__":
        pass
