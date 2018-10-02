# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 09:08:31 2018

@author: Greta
"""

# For choosing between inputs, and creating all possible combinations.Input to fuseStableImages

catDict = {'woman':'man','man':'woman','indoor':'outdoor','outdoor':'indoor',
           'woman':'man','man':'woman','indoor':'outdoor','outdoor':'indoor'} 
# 8 pairs, since I have 8 blocks for 1 run.
# NOT USED


catFacesDict = {'woman':'man','man':'woman','woman':'man','man':'woman'} 
catScenesDict = {'indoor':'outdoor','outdoor':'indoor', 'indoor':'outdoor','outdoor':'indoor'}
    
catFacesLst = [['woman','man'],['man','woman'],['woman','man'],['man','woman']] 
    
# When running a 8 stable blocks

random.shuffle(catFacesLst) #Shuffle randomly

for item in catFacesLst:
    domInput = item[0]
    lureInput = item[1]

catFacesLst.pop(0) # Delete the category pair that has just been used as input 

#%%
# Create inputs for fuseStableImages

# To generate the folders with images. Do not delete, but simply draw from them, and shuffle image order before
for k in range(0,8):
    aDom = catComb[k][0]
    aLure = catComb[k][1]
    nDom = catComb[k][2]
    nLure = catComb[k][3] 
    fuseStableImages(aDom, aLure, nDom, nLure)  
    
    images = [visual.ImageStim(win, image = data_path + '\stable_save\\no_%d.jpg' % trialIDs[idx_image]) for idx_image in range(len(trialIDs))]
    
#%%
# Run block code (works)
    
num_frames_period = np.array([660, 660, 660, 660, 3000, 3000, 3000, 3000, 3000, 3000, 3000]) #number of framer per each trial. Right now the same
num_frames_stimuli = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]) #number of framer per each trial. Right now the same


def runBlock(images):
    for trial in range(trials): # right now I have 11 trials
        trial_clock = core.Clock()
        
        for frameN in range(num_frames_period[trial]): # Time x frame rate
            
            if frameN == 0:
                print('frame0')
            
            
            if 0 <= frameN < num_frames_stimuli[trial]: 
                images[trial].draw()
            
                win.flip()    
    
    
    
    