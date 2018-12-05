# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:49:09 2018

@author: sofha
"""

from pylsl import StreamInfo, StreamOutlet,local_clock
import time
import random
import numpy as np

print ("Creating a new marker stream info...\n")
fs=500

info = StreamInfo('EnobioMock','EEG',32,fs,'float32','myuniquesourceid23444')

print("Opening an outlet...\n")
outlet =StreamOutlet(info)
infoMarker = StreamInfo('MyMarkerStream1','Markers',1,0,'int32','myuniquesourceid23443')

outletMarker =StreamOutlet(infoMarker)

# Get x values of the sine wave

timeSine        = np.arange(0, 2*np.pi, 2*np.pi/500);
# Amplitude of the sine wave is sine of a variable like time
amplitude   = np.sin(timeSine)
plt.plot(amplitude)


print("Sending data...\n")

mkr=0
c=0
TIME=np.zeros(30*60*fs)
vec = []
markers=[]
i=0
while (True):
    time.sleep(1/fs) #random.randint(0,3)
    markers.append(mkr)
    vec=np.random.rand(32)/10+amplitude[i]
    stamp=local_clock()
    TIME[c]=stamp
    outlet.push_sample(vec) 
    if i==499:
        i=0
        outletMarker.push_sample([mkr]) 
        print("Now sending: \t" + str(c)+"\n")
    else:
        i+=1
    #if (c%100)==0:
    #    print("Now sending: \t" + str(c)+"\n")
    c=c+1
    mkr = mkr+1