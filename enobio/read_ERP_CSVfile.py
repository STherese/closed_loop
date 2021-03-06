# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:43:01 2018

@author: nicped
"""

from scipy.signal import butter, lfilter, lfilter_zi


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65])/(256/2), btype='bandstop')

def preprocEEG(epoch):

    
#%%
glo=data_init(500,'test')
sample=sample_EEG
timestamp=timestamp_EEG
glo.filename='C:\\Users\\nicped\\Documents\\GitHub\\closed_loop\\logging\\EEGdata\\'+'data_'+glo.data_type+'_'+time.strftime("%H%M%S_%d%m%Y")+'.csv'
with open(glo.filename,'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(glo.header)
with open(glo.filename,'a',newline='') as csvfile:
#fieldnames=['name1','name2']
    writer = csv.writer(csvfile)
#time_s=np.array([timestamps]).T
    if len(sample)<=1:
        writer.writerow(np.append(np.array([sample]),np.array([timestamp])))
    else:
        writer.writerows(np.append(sample,np.array([timestamp]).T,axis=1))

#%%
import csv
import numpy as np
le=np.zeros(140000)
EEGdata=[]
c=0#data_EEG_134249_05122018.csv
sofha_dir='C:\\Users\\sofha\\Documents\\closed_loop\\closed_loop\\logging\\EEGdata\\'
mindreader_dir='C:\\Users\\nicped\Documents\\GitHub\\closed_loop\\logging\\EEGdata\\'
with open(sofha_dir+'data_EEG_142307_06122018.csv','r') as csvfile:
#fieldnames=['name1','name2']
    csvReader = csv.reader(csvfile)
    for row in csvfile:
        rownum=np.fromstring(row,dtype=float,sep=',')
        le[c]=(len(rownum))
        EEGdata.append(rownum)
        c+=1

marker=[]
marker_time=[]
c=0
with open(sofha_dir+'data_lsl_marker_142307_06122018.csv','r') as csvfile:
#fieldnames=['name1','name2']
    csvReader = csv.reader(csvfile)
    for row in csvfile:
        rownum=np.fromstring(row,dtype=float,sep=',')
        le[c]=(len(rownum))
        if len(rownum)>1:
            marker.append(rownum[0])
            marker_time.append(rownum[1])
        c+=1
        
EEGdata=EEGdata[2:]
eegdata=np.array(EEGdata)
#marker=marker[2:]
#marker_time=marker_time[2:]
#%% preprocess
from scipy import signal
fs=500
fs_new=100
num_samples=np.round(len(eegdata)/(fs/fs_new))
eegdata_resamp=signal.resample(eegdata, num_samples,window='boxcar')
#%%
import matplotlib.pyplot as plt
plt.plot(eegdata[:,-1]-eegdata[0,-1])

plt.plot(eegdata[:,7]-eegdata[0,7])
#%%
import scipy.io as sio
sio.savemat('eeg_dataERP.mat',{'eegdata':eegdata})