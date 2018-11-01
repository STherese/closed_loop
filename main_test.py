# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:12:52 2018

@author: Greta
"""

# Imports
from experiment_closed_loop_vol2_lenovo import * # or only some functions

global stableSaveCount

# manual data path
#data_path='C:\\Users\\Greta\\Documents\\GitHub\\closed_loop\\data'

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
catComb = [['man','woman','indoor','outdoor'],
           ['man','woman','outdoor','indoor'],
           ['woman','man','indoor','outdoor'],
           ['woman','man','outdoor','indoor'],
           ['indoor','outdoor','man','woman'],
           ['indoor','outdoor','woman','man'],
           ['outdoor','indoor','man','woman'],
           ['outdoor','indoor','woman','man']]

#global stableSaveCount 
#stableSaveCount = 1 

# Try to create different stable_save folders
# if beh session: initiate fuseStableImages2 8 times. 
# if NF session: initiate fuseStableImages2 4 times
# Delete the folders after a session - either manually or script 
# Remember :
# 1. Log which images are used in which order 
# 2. To randomly pick from the ones in e.g. face folder, so that different images are used each time
        # Currently is actually randomly samples from the folders, so that is pretty great. 
        # Can just use the order in stable_save (then it corresponds to the log file, too)

####### THIS SHOULD BE IN A SEPARATE PREP SCRIPT #######

# Create the fused images to display (for stable images)
for k in list(range(0,numStableBlocks)):
    aDom = catComb[k][0]
    aLure = catComb[k][1]
    nDom = catComb[k][2]
    nLure = catComb[k][3] 
    fuseStableImages2(aDom, aLure, nDom, nLure)  
    
    
    
# Find the stable_save folders that have been generated
nameStableCats = findCategories(data_path + '\stable\\') #use this as input to InitializeStableBlock
    
for i in list(range(0,numStableBlocks)):   
    folderName = nameStableCats[i]
    initializeStableBlock(folderName)
    
    if i == numStableBlocks-1:
        closeWin()
    
    
    










# When the preprocessing script is finished, and everything is generated:
    # 1. del stableSaveCount OR stableSaveCount = 1
    # 2. delete the generated stable_save folders 