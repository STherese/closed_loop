"""
Created on Tue Sep 11 09:13:53 2018

@author: gretatuckute

Reads multi-channel time series from EEG (Enobio needs to be connected via wifi)

This version runs without activating LSL LabRecorder.exe (NIC2.0 running - with LSL outlet)
"""

# Imports 
from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np
import bci_workshop_tools as BCIw  # Our own functions for the workshop


streams = resolve_stream('type', 'EEG') # Searches for EEG stream

print("Number of streams found: %s " % len(streams))

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

sample, timestamp = inlet.pull_sample() # pull_sample() gives the most recent sample of a stream

# Checks
print(sample)
print(len(sample))
print(timestamp)

t0 = [local_clock()] * inlet.channel_count

# eeg_time_correction = inlet.time_correction()

# Get the stream info, description, sampling frequency, number of channels
info = inlet.info()
description = info.desc()
fs = int(info.nominal_srate())
n_channels = info.channel_count()

# Get names of all channels
ch = description.child('channels').first_child()
ch_names = [ch.child_value('label')]
for i in range(1, n_channels):
    ch = ch.next_sibling()
    ch_names.append(ch.child_value('label'))
    
# Setting parameters
buffer_length = 15

# Length of the epochs used to compute the FFT (in seconds)
epoch_length = 1

# Amount of overlap between two consecutive epochs (in seconds)
overlap_length = 0.8

# Amount to 'shift' the start of each next consecutive epoch
shift_length = epoch_length - overlap_length

# Index of the channel (electrode) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
index_channel = [0, 1, 2, 3]
# Name of our channel for plotting purposes
ch_names = [ch_names[i] for i in index_channel]
n_channels = len(index_channel)

feature_names = BCIw.get_feature_names(ch_names)


"""3. INITIALIZE BUFFERS """

# Initialize raw EEG data buffer (for plotting)
eeg_buffer = np.zeros((int(fs * buffer_length), n_channels))
filter_state = None  # for use with the notch filter

# Compute the number of epochs in "buffer_length" (used for plotting)
n_win_test = int(np.floor((buffer_length - epoch_length) /
                          shift_length + 1))

# Initialize the feature data buffer (for plotting)
feat_buffer = np.zeros((n_win_test, len(ch_names)))

# Initialize the plots
plotter_eeg = BCIw.DataPlotter(fs * buffer_length, ch_names, fs)
plotter_feat = BCIw.DataPlotter(n_win_test, feature_names,
                                1 / shift_length)

""" 3. GET DATA """

# The try/except structure allows to quit the while loop by aborting the
# script with <Ctrl-C>
print('Press Ctrl-C in the console to break the while loop.')

try:
    # The following loop does what we see in the diagram of Exercise 1:
    # acquire data, compute features, visualize raw EEG and the features
    while True:

        """ 3.1 ACQUIRE DATA """
        # Obtain EEG data from the LSL stream
        eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(shift_length * fs))

        # Only keep the channel we're interested in
        ch_data = np.array(eeg_data)[:, index_channel]

        # Update EEG buffer
        eeg_buffer, filter_state = BCIw.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

        """ 3.2 COMPUTE FEATURES """
        # Get newest samples from the buffer
        data_epoch = BCIw.get_last_data(eeg_buffer,
                                        epoch_length * fs)

        # Compute features
        feat_vector = BCIw.compute_feature_vector(data_epoch, fs)
        feat_buffer, _ = BCIw.update_buffer(feat_buffer,
                                            np.asarray([feat_vector]))

        """ 3.3 VISUALIZE THE RAW EEG AND THE FEATURES """
        plotter_eeg.update_plot(eeg_buffer)
        plotter_feat.update_plot(feat_buffer)
        plt.pause(0.00001)

except KeyboardInterrupt:

    print('Closing!')





# Can also read time series from LSL in chunks
# chunk, timestamps = inlet.pull_chunk()

#while True:
    # get a new sample 
    # sample, timestamp = inlet.pull_sample()
    # print("Timestamp: \t %0.5f\n Sample: \n %s\n\n" %(timestamp,   sample))
#print(timestamp, sample)

# Implement local_clock()
# LSL time stamps the data by push_sample() or to call the local_clock() function to read out the LSL clock, and then pass that in
