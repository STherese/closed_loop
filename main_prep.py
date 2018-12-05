# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
import os
os.chdir('C:\\Users\\Greta\\Documents\GitHub\closed_loop')
# os.chdir('C:\\Users\\nicped\\Documents\GitHub\closed_loop')


from experiment_closed_loop_vol23 import createIndices, closeWin
import numpy as np
import random

closeWin()
############### Global variables ###############
global stableSaveCount
global imgIdx

subjID = 4
numRuns = 1 


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

for run in list(range(0,numRuns)):
    for index in randCat:
        aDom = catComb1[index][0]
        aLure = catComb1[index][1]
        nDom = catComb1[index][2]
        nLure = catComb1[index][3]
        createIndices(aDom, aLure, nDom, nLure, subjID)
        

#expMode = 'beh'
#
## Initializing variables based on type of session
#if expMode == 'beh':
#    numStableBlocks = 2 
#
#if expMode == 'nf':
#    numStableBlocks = 4
