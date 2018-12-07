# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
import os
os.chdir('C:\\Users\\Greta\\Documents\GitHub\closed_loop')
# os.chdir('C:\\Users\\nicped\\Documents\GitHub\closed_loop')


from experiment_closed_loop_vol23 import *
import numpy as np
import random
import time

############### Global variables ###############
global stableSaveCount
global imgIdx


#%%
subjID = 4
numRuns = 1

blockLen = 1
numBlocks = 2      
runLen = numBlocks * blockLen # Should be 8 * 50
    
for run in list(range(0,numRuns)):
    for ii in list(range(0,runLen)):
        if ii % blockLen == 0: # Correct: ii % 50
            runFixProbe(log_path + '\\createIndices_' + str(subjID) + '.csv')
        fuseImage(log_path + '\\createIndices_' + str(subjID) + '.csv')    
        
        if ii == runLen-1:
            # win.close()
            # Inset break?
            runBreak(300,'break time, yay, you rock!!')
            print('Finished a run of length: ' + str(runLen))
            print('Finished run no.: ' + str(run + 1))
        
    if run == numRuns-1:  
        runBreak(900,'Finito!')
        print('No. total trials finished: ' + str(runLen * numRuns - 1))
        print('Block length (no. images in each block): ' + str(blockLen))
        print('Total number of blocks finished: ' + str(numBlocks))
        print('No. of runs finished: ' + str(numRuns))     
        closeWin()     
          