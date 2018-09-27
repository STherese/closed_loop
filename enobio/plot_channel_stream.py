# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 2018

@author: gretatuckute

Reads multi-channel time series from EEG (Enobio needs to be connected via wifi). Plots a single channel sample activity with a small delay. 

FIRST CELL: uses pull_sample() 
SECOND CELL: uses pull_chunk()
THIRD CELL: Appends 

This version runs without activating LSL LabRecorder.exe (NIC2.0 running - with LSL outlet)


"""
# pull_sample() samples quickly, does not need time.sleep (pull_chunk() needs this)

# Imports 
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
import time

streams = resolve_stream('type', 'EEG') # Searches for EEG stream
inlet = StreamInlet(streams[0])
sample, timestamp = inlet.pull_sample()

plt.ion()
fig = plt.figure()
axes = fig.add_subplot(111)

axes.set_autoscale_on(True)
axes.autoscale_view(True,True,True)

l, = plt.plot([],[], 'r-')
plt.xlabel('timestamp')
plt.ylabel('EEG channel 1 sample')
plt.title('EEG test')

xdata = [timestamp]
ydata = [sample[0]]

# while True:
for i in range(0,1000):
    # FOR PULL CHUNK - does not make sense
#    eeg_data, timestamp = inlet.pull_chunk(timeout=2.0, max_samples=32)
#    eeg = np.array(eeg_data)
#    ch1 = eeg[:,1]
#    ch1_lst = np.ndarray.tolist(ch1)
    
    # FOR PULL SAMPLE
    sample, timestamp = inlet.pull_sample()
    ch1 = sample[0]
    
    # time.sleep(0.1)
    ydata.append(ch1)
    xdata.append(timestamp)
    l.set_data(xdata,ydata)
    # print(xdata) # Prints the precise time stamp used for x axis
    # print(ydata) # Prints the ch1 sample from EEG stream
    axes.relim()
    axes.autoscale_view(True,True,True)
    plt.draw()
    plt.pause(0.001)
    
inlet.close_stream()
#%% 
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

#%% 
    
from pylsl import StreamInlet, resolve_stream
import numpy as np
import matplotlib.pyplot as plt
import time

streams = resolve_stream('type', 'EEG') # Searches for EEG stream
inlet = StreamInlet(streams[0])

samplelst=[]
timelst=[]

for j in range(0,1000):
    sample, timestamp = inlet.pull_chunk(timeout=1.0, max_samples=164)
    samplelst.append(sample)
    timelst.append(timestamp)
   # time.sleep(0.1)
    
    
    