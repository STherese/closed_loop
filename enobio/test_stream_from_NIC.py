# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:07:03 2018

@author: nicped
"""

from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np
import matplotlib.pyplot as plt
import time

streams = resolve_stream('type', 'EEG') # Searches for EEG stream
print("Number of streams found: %s " % len(streams))

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
sample, timestamp = inlet.pull_sample()
streamMarkers = resolve_stream('type', 'Markers') # Searches for EEG stream
inletM0=StreamInlet(streamMarkers[0])
inletM1=StreamInlet(streamMarkers[1])
sampleMarker1, timestampMarker1 = inletM1.pull_sample()
sampleMarker0, timestampMarker0 = inletM0.pull_sample()

#%%
info = inlet.info()
description = info.desc()
fs = int(info.nominal_srate())
n_channels = info.channel_count()

eeg_figure, = plt.plot([],[])
plt.ion()
plt.axis([timestamp-50,timestamp+50,-30000,30000])


def update_eeg(eeg_figure, t, new_data):
    eeg_figure.set_xdata(np.append(eeg_figure.get_xdata(),t))
    eeg_figure.set_ydata(np.append(eeg_figure.get_ydata(),new_data))    
    plt.draw()
    plt.pause(0.1)

for i in range(0,250):
    eeg_data, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=32)
    eeg = np.array(eeg_data)
    ch1 = eeg[:,1]
    
    time.sleep(0.08)
    update_eeg(eeg_figure,timestamps,ch1)