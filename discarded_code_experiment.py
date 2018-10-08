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
    
#%% Another version
                
frame_rate = 60
stimuli_time = 1 # in seconds
trials = 11 # images in block # 50

num_frames_period = np.array([660, 780, 780, 780, 3000, 3000, 3000, 3000, 3000, 3000, 3000]) #number of framer per each trial. Right now the same
num_frames_stimuli = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]) #number of framer per each trial. Right now the same

num_frames_period = num_frames_period.astype('int') # Denotes how long the trial periods are, e.g. 11 imgs x 1 s x 60 Hz = 660 frames
num_frames_stimuli = num_frames_stimuli.astype('int')
                
                
def runBlock(images):
    for trial in range(trials): # right now I have 11 trials
        trialClock = core.Clock()
#        t = trialClock.getTime()
#        timeLst.append(t)
        #fixation.draw()
        
        for frameN in range(num_frames_stimuli[trial]): # Time x frame rate            
            if frameN >= 0:
                #print('frame0') #TEXT
                probeword_text.draw()
            
            win.flip()
                
        for frameN in range(num_frames_period[trial]): # Time x frame rate            

            #if num_frames_stimuli[trial] <= frameN: 
            if 0 <= frameN < num_frames_stimuli[trial]: 
                #images[trial].draw() #CURRENTLY ONLY DRAWING ONE IMAGE 

                #fixation.draw()
                # timeLst.append(t)
                
                win.flip()
                
#%% Works correctly, with small bugs
                
    for frameN in range(0,660): # 660 is 60*11, but it should be 60*50
        if frameN % 60 == 0:
            counter += 1
            for newframe in range(0,60):
                if newframe >= 0:
                    print(newframe)
                    images[counter].draw()
                
            
                win.flip()
        
#%% Initial attempt of fuseStableImages (might be used for creating fuseImages in NF blocks)
                
def fuseStableImages(aDom, aLure, nDom, nLure): #make arguments that decide which categories to take and input in create random imgs
    """Returns a fused image with alpha 0.5 based on a chosen attentive category, and a chosen unattentive category.
        Saves the fused images in a folder named after the attentive category.
    
    # Arguments:
        Names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks
        
        Save the images? Delete them afterwards
        Save the image IDs to a log file
    
    """    

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure)    
    
#    aCatImages, aCats = createRandomImages(dom='woman',lure='man') # 50 attend category images
#    nCatImages, nCats = createRandomImages(dom='outdoor',lure='indoor')
    
    # df = pd.DataFrame(columns=['img1', 'img2']) # Initializing log pd df file

    imageCount = 0
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        fusedImage = Image.blend(background, foreground, .5)
    
        fusedImage.save(data_path + '\stable_save\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        fusedImage.close()
    
        imageCount += 1
        
        # Savo to pd dataframe to write log file
        # df.loc[i] = [aCatImages[i], nCatImages[i]] # need to index differently, to add across calls
    # incrCount += 1
    
    #df.loc[i + incrCount * len(aCatImages)] = ['NEW BLOCK'] # Writes to df in order to not overwrite 
    #df.to_csv(data_path + '\logs' '\log_fuseStableImages.csv')
    
        with open(data_path + '\logs' '\log_fuseStableImages.csv', 'a') as logFile: #dont open it each time
            logFile.write(aCatImages[i] + nCatImages[i])
            logFile.write('\n') # JUST FOR TESTING WHETHER STUFF WORKS. MAKE INTO DF LATER
            
        
    with open(data_path + '\log_fuseStableImages.csv', 'a') as logFile:
        logFile.write('NEW BLOCK') 
        
    print('Created %d fused images in stable_save with alpha 0.5.\n' % (len(aCatImages)))
    