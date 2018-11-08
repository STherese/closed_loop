# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
from experiment_closed_loop_vol21 import * # or only some functions
import numpy as np
import random


global stableSaveCount

# Choose between either behavioral experimental session, or NF session
# OR create two different main scripts? 
expMode = 'beh'
# expMode = nf

# Initializing variables based on type of session
if expMode == 'beh':
    numStableBlocks = 2 # 8 or 16, but testing with 2

if expMode == 'nf':
    numStableBlocks = 4

# Initializing possible combinations for beh sess
catComb = [['male','female','indoor','outdoor'],
           ['male','female','outdoor','indoor'],
           ['female','male','indoor','outdoor'],
           ['female','male','outdoor','indoor'],
           ['indoor','outdoor','male','female'],
           ['indoor','outdoor','female','male'],
           ['outdoor','indoor','male','female'],
           ['outdoor','indoor','female','male']]

catComb1 = [['male','female','indoor','outdoor'],
            ['indoor','outdoor','male','female']]

catComb2 = [['male','female','outdoor','indoor'],
            ['outdoor','indoor','male','female']]

catComb3 = [['female','male','indoor','outdoor'],
            ['indoor','outdoor','female','male']]

catComb4 = [['female','male','outdoor','indoor'],
            ['outdoor','indoor','female','male']]

####### THIS SHOULD BE IN A SEPARATE PREP SCRIPT #######

# Create fused images to display for 1 block (8 runs)

# I want to create 4 blocks with catComb1[0] and 4 blocks with catComb1[1]

# Create list with 8 random entries of 4x0 and 4x1
randLst = [0,0,0,0,1,1,1,1]
randCat  = random.sample(randLst, 8)

# Generates the chosen combination for a subj
for index in randCat:
    aDom = catComb1[index][0]
    aLure = catComb1[index][1]
    nDom = catComb1[index][2]
    nLure = catComb1[index][3]
    fuseStableImages(aDom, aLure, nDom, nLure)


for k in list(range(0,numStableBlocks)):
    aDom = catComb[k][0]
    aLure = catComb[k][1]
    nDom = catComb[k][2]
    nLure = catComb[k][3] 
    fuseStableImages(aDom, aLure, nDom, nLure)  
    
    
    
# Find the stable_save folders that have been generated
nameStableCats = findCategories(stable_path) #use this as input to InitializeStableBlock
    
for i in list(range(0,numStableBlocks)):   
    folderName = nameStableCats[i]
    initializeStableBlock(folderName)
    
    if i == numStableBlocks-1:
        closeWin()
    
    
    










# When the preprocessing script is finished, and everything is generated:
    # 1. del stableSaveCount OR stableSaveCount = 1
    # 2. delete the generated stable_save folders 