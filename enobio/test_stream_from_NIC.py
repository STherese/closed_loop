# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 13:07:03 2018

@author: nicped
"""

from pylsl import StreamInlet, resolve_stream, local_clock
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

streams = resolve_stream('type', 'EEG') # Searches for EEG stream
print("Number of streams found: %s " % len(streams))

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
sample, timestamp = inlet.pull_sample()
#streamMarkers = resolve_stream('type', 'Markers') # Searches for EEG stream
#inletM0=StreamInlet(streamMarkers[0])
#inletM1=StreamInlet(streamMarkers[1])
#sampleMarker1, timestampMarker1 = inletM1.pull_sample()
#sampleMarker0, timestampMarker0 = inletM0.pull_sample()

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
#%%
    
class data:
    def __init__(self, fs,filename=None):
        self.fs,self.filename = fs,filename
glo=data(500)
#glo.options.filename=time.strftime("%H%M%S_%d%m%Y")
def save_data(glo,eeg,timestamps):
    #if exist(glo.options.filename)
    if glo.filename==None:
        glo.filename='data_'+time.strftime("%H%M%S_%d%m%Y")+'.csv'
        with open(glo.filename,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['time stamp','marker'])
    with open(glo.filename,'a') as csvfile:
        #fieldnames=['name1','name2']
        writer = csv.writer(csvfile)
        #writer.writeheader()
        writer.writerow([eeg,timestamps])
    return glo
glo=save_data(glo,51,'310')
glo=save_data(glo,51,'313333')
#%%

#### 		Read from Enobio LSL Markers		####

 # The stream_name must be the same as in NIC/COREGUI.
 # The streams vector gathers all the streams available.
 # If the NIC stream is inside this vector, the element index is saved in the index variable.
 # The stream inlet attempts to connect to the NIC stream.
 # If the stream has not been found within the available streams, the scripts raises an error and stops.
 # If not, the script starts retrieving data from the NIC stream.

from pylsl import StreamInlet, resolve_stream
import csv
import time
glo=data(500)
stream_name_Enobio = 'Enobio1-Markers'
streams = resolve_stream('type', 'Markers')
stream_name_lsl = 'MyMarkerStream3'
 
for i in range (len(streams)):
    if (streams[i].name() == stream_name_Enobio):
        index_enobio = i
        print ("NIC stream available")
    if (streams[i].name() == stream_name_lsl):
        index_lsl = i
        print ("lsl stream available")

#print ("Connecting to NIC stream... \n")
inlet_enobio = StreamInlet(streams[index_enobio])   
inlet_lsl = StreamInlet(streams[index_lsl])   

#except NameError:
#	print ("Error: NIC stream not available\n\n\n")
    


while True:
    sample, timestamp = inlet_enobio.pull_sample()#_chunk(timeout=0,max_samples=500)
    sample_lsl, timestamp_lsl = inlet_lsl.pull_sample()
    #print("Timestamp: \t %0.5f\n Sample: \t %s\n\n" %(timestamp,   sample))
    glo=save_data(glo,timestamp,sample)
    glo=save_data(glo,timestamp_lsl,sample_lsl)
    
#%%
import matplotlib.pyplot as plt
import numpy as np
Timestamps=[]
Markers=[]
with open(glo.filename,'r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        if row:
            Timestamps.append(row[0])
            Markers.append(row[1])
        #print(row)
        #rowNr=rowNr+1
lsl_markers=Markers[2::2]
enobio_markers=Markers[3::2]
lsl_time=Timestamps[2::2]
lsl_time=np.asarray([float(s) for s in lsl_time])
enobio_time=Timestamps[3::2]
enobio_time=np.asarray([float(s) for s in enobio_time])
plt.figure(3)
plt.plot(lsl_markers)
plt.plot(enobio_markers,'--')
plt.figure(4)
plt.plot(lsl_time)
plt.plot(enobio_time,'--')
plt.figure(5)
plt.plot(lsl_time[0:-1]-enobio_time)
plt.figure(6)
plt.plot(np.diff(lsl_time))
plt.plot(np.diff(enobio_time))
#plt.plot(enobio_time,'--')
#%%
lsl_time=[float(s) for s in lsl_time]