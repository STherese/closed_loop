# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:13:53 2018

@author: gretatuckute

Reads multi-channel time series from EEG (Enobio needs to be connected via wifi)

This version runs without activating LSL LabRecorder.exe (NIC2.0 running - with LSL outlet)

This code plots the most recent samples for a chosen channel - BUT plt loads while it does it, and thus it is not visible. But it works correctly.
"""

# Imports 
from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np
import matplotlib.pyplot as plt
import time


streams = resolve_stream('type', 'EEG') # Searches for EEG stream

print("Number of streams found: %s " % len(streams))

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

sample, timestamp = inlet.pull_sample()
t0 = [local_clock()] * inlet.channel_count


print(sample)
print(len(sample))
# print(t0)

info = inlet.info()
description = info.desc()
fs = int(info.nominal_srate())
n_channels = info.channel_count()

eeg_figure, = plt.plot([],[])

#%% 
#eeg_data, timestamp = inlet.pull_chunk()

#ch = np.array(eeg_data)

#ch1=ch[:,1]

def update_eeg(eeg_figure, t, new_data):
    eeg_figure.set_xdata(np.append(eeg_figure.get_xdata(),t))
    eeg_figure.set_ydata(np.append(eeg_figure.get_ydata(),new_data))    
    plt.axis([timestamp-50,timestamp+50,-40000,40000])
    plt.draw()

for i in range(0,250):
    eeg_data, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=32)
    eeg = np.array(eeg_data)
    # ch1=eeg[:,1]
    
    time.sleep(0.1)
    update_eeg(eeg_figure,timestamps,eeg[:,1])

    
#    
#    
#%%

#plt.ion()
#
#fig.show()
#fig.canvas.draw()

# Plots a channel for e.g. 8 time stamps 
#for i in range(0,100):
#    eeg_data, timestamp = inlet.pull_chunk()
#    ch1=ch[:,1]
#    ax.clear()
#    ax.plot(ch1)
#    fig.canvas.draw()
    
    




#while True:
#    eeg_data, timestamp = inlet.pull_chunk()
#    plt.plot(eeg_data)
#    plt.show()


# Can also read time series from LSL in chunks
# chunk, timestamps = inlet.pull_chunk()


#while True:
    # get a new sample 
#    sample, timestamp = inlet.pull_sample()
#print(timestamp, sample)

# If I want to check the most recent sample of a stream, try pull_sample()

# Implement local_clock()

# LSL time stamps the data by push_sample() or to call the local_clock() function to read out the LSL clock, and then pass that in