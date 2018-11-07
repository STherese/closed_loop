# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 22:26:09 2018

@author: Greta
"""

import os 
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\Greta\\Documents\\GitHub\\decoding\\pckl\\SVM_pairs\\5')

ef5 = np.load('effect_smap5.npy')
mean5 = np.load('mean_smap5.npy')
std5 = np.load('std_smap5.npy')


channel_vector = ['P7','P4','Cz','Pz','P3','P8','O1','O2','T8','F8','C4','F4',
                  'Fp2','Fz','C3','F3','Fp1','T7','F7','Oz','PO3','AF3','FC5',
                  'FC1','CP5','CP1','CP2','CP6','AF4','FC2','FC6','PO4']

time_vector = ['-100','0','100','200','300','400','500']

plt.matshow(ef5)
plt.xlabel('Time (ms)')
plt.xticks(np.arange(0,60,10),time_vector)
plt.yticks(np.arange(32),channel_vector)
plt.ylabel('Electrode number ')
bounds=[0,1,2,3,4]
plt.colorbar()
#plt.yticks(np.arange(len(gamma_range)), gamma_range)
#plt.xticks(np.arange(len(C_range)), C_range, rotation=45)
plt.title('Effect size')
plt.show()

