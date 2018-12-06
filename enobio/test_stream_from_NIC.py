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
import csv
class data_init:
    def __init__(self, fs,data_type,filename=None):
        self.fs,self.filename,self.data_type = fs,filename,data_type
        
glo=data_init(500,'test')
#glo.options.filename=time.strftime("%H%M%S_%d%m%Y")
def save_data(data,sample,timestamp):
    #if exist(glo.options.filename)
    if data.filename==None:
        data.filename='C:\\Users\\nicped\\Documents\\GitHub\\closed_loop\\logging\\EEGdata\\'+'data_'+data.data_type+'_'+time.strftime("%H%M%S_%d%m%Y")+'.csv'
        with open(data.filename,'w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data.header)
    with open(data.filename,'a',newline='') as csvfile:
        #fieldnames=['name1','name2']
        writer = csv.writer(csvfile)
        #time_s=np.array([timestamps]).T
        if len(sample)<=1:
            writer.writerow(np.append(np.array([sample]),np.array([timestamp])))
        else:
            writer.writerows(np.append(sample,np.array([timestamp]).T,axis=1))
    return data

def clear_stream(inlet):
    sample0, timestamp0 = inlet.pull_chunk(max_samples=1500)
#glo=save_data(glo,51,'310')
#glo=save_data(glo,51,'313333')
#%%
NOTCH_b, NOTCH_a = butter(4, np.array([45, 55])/(256/2), btype='bandstop')
from scipy.signal import butter, lfilter, lfilter_zi
def nothch_filter_data(data,NOTCH_A=NOTCH_a,NOTCH_B=NOTCH_a):
    if filter_state is None:
                filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),(data.shape[1], 1)).T
                new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,zi=filter_state)
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
import numpy as np


time.sleep(1)
stream_name_Enobio = 'Enobio1-Markers'
streams = resolve_stream('type', 'Markers')
stream_name_lsl = 'PsychopyExperiment3'#'MyMarkerStream1'
enobio_avail=0
lsl_avail=0
look_for_enobio=1
look_for_enobio_EEG=1
max_buf=2
fs=500
import warnings

if look_for_enobio_EEG:
    streamsEEG = resolve_stream('type', 'EEG')
    inlet_EEG=StreamInlet(streamsEEG[0],max_buflen=max_buf)
    store_EEG=data_init(500,'EEG')
    store_EEG.header=['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4','Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5','FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4','Timestamp']
for i in range (len(streams)):
    if look_for_enobio:
        if (streams[i].name() == stream_name_Enobio):
            index_enobio = i
            print ("NIC stream available")
            inlet_enobio = StreamInlet(streams[index_enobio]) 
            enobio_avail=1
            store_enobio_marker=data_init(500,'enobio_marker')
            store_enobio_marker.header=['Marker','Timestamp']
    if (streams[i].name() == stream_name_lsl):
        index_lsl = i
        print ("lsl stream available")
        inlet_lsl = StreamInlet(streams[index_lsl])
        lsl_avail=1
        store_lsl_marker=data_init(500,'lsl_marker')
        store_lsl_marker.header=['Marker','Timestamp']

excess_EEG=np.zeros(32)
excess_time_EEG=[]
pull_interval=1
cnt=1
look_for_triggers=1
run_exp=0
C=1
while True:
    t1=time.clock()
    if look_for_triggers:
        if enobio_avail:       
            if cnt ==1:
                clear_stream(inlet_enobio)
            sample, timestamp = inlet_enobio.pull_chunk()#_chunk(timeout=0,max_samples=500)
            sample=np.asarray(sample)
            timestamp =np.asarray(timestamp) 
            store_enobio_marker=save_data(store_enobio_marker,sample,timestamp)
        if lsl_avail:
            if cnt ==1:
                clear_stream(inlet_lsl)
            sample_lsl, timestamp_lsl = inlet_lsl.pull_chunk()
            sample_lsl=np.asarray(sample_lsl)
            timestamp_lsl =np.asarray(timestamp_lsl) 
            store_lsl_marker=save_data(store_lsl_marker,sample_lsl,timestamp_lsl)
    if look_for_enobio_EEG:
        if cnt ==1:
            clear_stream(inlet_EEG)
        sample_EEG, timestamp_EEG = inlet_EEG.pull_chunk()#_chunk(timeout=0,max_samples=500)   
        sample_EEG=np.asarray(sample_EEG)
        timestamp_EEG =np.asarray(timestamp_EEG) 
        store_EEG=save_data(store_EEG,sample_EEG,timestamp_EEG)
        if excess_EEG.any():
            sample_EEG=np.concatenate((excess_EEG, sample_EEG),axis=0)
            timestamp_EEG=np.concatenate((excess_time_EEG, timestamp_EEG),axis=0)
    
    
    if run_exp and sample_lsl.any(): # find stimuli onset in EEG
        if len(sample_lsl)>1:    # should be changed to enobio marker    
            print("Warning. More than two trigger points recovered, using most recent one")
            sample_lsl=sample_lsl[-1]
            timestamp_lsl=timestamp_lsl[-1]
            
        i_start=np.argmin(np.abs(timestamp_lsl-timestamp_EEG))
        avail_samples=(len(timestamp_EEG)-i_start)
        
        if avail_samples>=fs:
            C+=1
            epoch=sample_EEG[i_start:i_start+fs-1] # 32 x 500 
            #plt.figure(C)
            #plt.plot(epoch)
            #do preprocess and classify
            #...
            #...
            #...
            print('Classifying')
            
            look_for_triggers=1
            excess_EEG=sample_EEG[i_start+fs:]
            excess_time_EEG=timestamp_EEG[i_start+fs:]
            t2=time.clock()
            if (t2-t1)<pull_interval: 
                t2=time.clock()   
                #print((t2-t1))
                time.sleep(max(pull_interval-(t2-t1),0))# Time to next epoch should be ready  
        else:
            print("Warning. Not enough EEG samples available")
            time.sleep(pull_interval-avail_samples/fs)
            look_for_triggers=0
            excess_EEG=sample_EEG
            excess_time_EEG=timestamp_EEG

    elif not sample_lsl.any():
        print("Warning. No trigger points recovered")
        time.sleep(0.1)
        look_for_triggers=1
        excess_EEG=sample_EEG
        excess_time_EEG=timestamp_EEG
    else:
        time.sleep(1)
    
    cnt+=1
    #print("Timestamp: \t %0.5f\n Sample: \t %s\n\n" %(timestamp,   sample))
    
#%%
#glo.filename='C:\\Users\\sofha\\data_105337_09112018.csv'#'C:\\Users\\sofha\\data_083922_09112018.csv
TimestampsEEG=np.zeros(10005)
channels=np.zeros((10005,32))
count=0
cc=0
with open(gloEEG.filename,'r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        cc=1+cc
        if cc>1:
            if len(row)>5:
                TimestampsEEG[count]=float(row[32])
                channels[count,:]=np.array((row[0:32]))
                count=count+1
               # a=np.array(row)
            #channels.append(row[0:31])
        #print(row)
        #rowNr=rowNr+1
#%%
clear_stream(inlet_EEG)
sample_EEG1, timestamp_EEG1 = inlet_EEG.pull_chunk()
sample_EEG2, timestamp_EEG2 = inlet_EEG.pull_chunk()
#%%
import matplotlib.pyplot as plt
import numpy as np
glo=data(500)
#glo.filename='C:\\Users\\sofha\\Documents\\closed_loop\\closed_loop\\data_105734_08112018.csv'
glo.filename='C:\\Users\\sofha\\data_143139_09112018.csv'#'C:\\Users\\sofha\\data_083922_09112018.csv
Timestamps=[]
Markers=[]
with open(glo.filename,'r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        if row:
            Timestamps.append(row[1])
            Markers.append(row[0])
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
plt.plot(lsl_time[:-1]-enobio_time)
plt.figure(6)
plt.plot(np.diff(lsl_time))
plt.plot(np.diff(enobio_time))
#plt.plot(enobio_time,'--')
#%%
M=[]
for m in range(len(enobio_markers)):
    M.append(np.argmin(np.abs(TimestampsEEG-enobio_time[m])))
plt.figure(11)
plt.plot(TimestampsEEG[0:7809],channels[0:7809,7])
plt.plot(enobio_time,np.ones(len(enobio_time)),'x')
plt.figure(12)
plt.plot(enobio_time-TimestampsEEG[M],'x-')
#%%
#sent_lsl=np.load('C:\\Users\\sofha\\Documents\\closed_loop\\closed_loop\\enobio\\tmpcyu29da5')#('C:\\Users\\sofha\\Documents\\closed_loop\\closed_loop\\Python_Scripts\\tmp6so82iut')
sent_lsl=[]
with open('C:\\Users\\sofha\\time_stamps_from_lsl.csv','r') as csvfile:
    reader=csv.reader(csvfile)
    for row in reader:
        sent_lsl.append(row)
sent_lsl=sent_lsl[0]
sent_lsl_time=np.asarray([float(s) for s in sent_lsl])
#%%

