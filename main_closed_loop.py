# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
import os
os.chdir('C:\\Users\\Greta\\Documents\GitHub\closed_loop')
# os.chdir('C:\\Users\\nicped\\Documents\GitHub\closed_loop')


from experiment_closed_loop_vol22 import *
import numpy as np
import random

############### Global variables ###############
global stableSaveCount
global imgIdx

subjID = 1

# Choose between either behavioral experimental session, or NF session
# OR create two different main scripts? 
expMode = 'beh'
# expMode = nf

# Initializing variables based on type of session
if expMode == 'beh':
    numStableBlocks = 2 # 8 or 16, but testing with 2

if expMode == 'nf':
    numStableBlocks = 4

catComb1 = [['male','female','indoor','outdoor'],
            ['indoor','outdoor','male','female']]

catComb2 = [['male','female','outdoor','indoor'],
            ['outdoor','indoor','male','female']]

catComb3 = [['female','male','indoor','outdoor'],
            ['indoor','outdoor','female','male']]

catComb4 = [['female','male','outdoor','indoor'],
            ['outdoor','indoor','female','male']]

############### PREP ONLY: INDICES FOR FUSING IMGS ###############
# Create fused images to display for 1 block (8 runs)
# I want to create 4 blocks with catComb1[0] and 4 blocks with catComb1[1]

# Create list with 8 random entries of 4x0 and 4x1
randLst = [0,0,0,0,1,1,1,1] # Currently for just one block 
randCat  = random.sample(randLst, 8)

# Generates the chosen combination for a subj
# Run this loop the no. of times corresponding to the total number of blocks that the subj has to complete (beh + nf)

# Randomly assign 4 subjs to catComb1, 4 subjs to catComb2 etc.

numRuns = 3 # 2 + 2 + 1 + ??? (nf day)

for run in list(range(0,numRuns)):
    for index in randCat:
        aDom = catComb1[index][0]
        aLure = catComb1[index][1]
        nDom = catComb1[index][2]
        nLure = catComb1[index][3]
        createIndices(aDom, aLure, nDom, nLure, subjID)


############### RUN ###############
# testing whether fuseImage and runImage works
blockLen = 11# 50
numBlocks = 4 # 8      
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
          

    



############### RUNNING SAVED IMAGES IN BLOCKS (vol2.0) ############### 
    
# Find the stable_save folders that have been generated
#nameStableCats = findCategories(stable_path) #use this as input to InitializeStableBlock
#    
#for i in list(range(0,numStableBlocks)):   
#    folderName = nameStableCats[i]
#    initializeStableBlock(folderName)
#    
#    if i == numStableBlocks-1:
#        closeWin()
#    
    
    










# When the preprocessing script is finished, and everything is generated:
    # 1. del stableSaveCount OR stableSaveCount = 1
    # 2. delete the generated stable_save folders 