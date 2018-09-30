# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:08:54 2018

@author: gretatuckute
"""
# Imports
import os
print(os.getcwd())
os.chdir('C:\\Users\\Greta\\Documents\GitHub\closed_loop')
print(os.getcwd())

from PIL import Image
import os # Can use os.getcwd() to check current working directory
import random
from random import sample
import sys  
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from psychopy import gui, visual, core, data, event, monitors
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
import time 
import numpy as np
from settings import path_init
import csv
import pandas as pd

data_path = path_init()

# Global variables

global incrCount # This needs to go in main
incrCount = 0 # This needs to go in main

############### EXPERIMENT FUNCTIONS #################

def findCategories(directory):
    """Returns the overall category folders in /data/.
    
    # Arguments
        directory: Path to the data (use data_path)
        
    # Returns
        found_categories: List of overall categories
    """
    
    found_categories = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            found_categories.append(subdir)
    return found_categories

def recursiveList(directory):
    """Returns list of tuples. Each tuple contains the path to the folder 
    along with the names of the images in that folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        list to use in findImages to iterate through folders
    """
    follow_links = False
    return sorted(os.walk(directory, followlinks=follow_links), key=lambda tpl: tpl[0])

def findImages(directory):
    """Returns the number of samples in and categories in a given folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        noImages: Total number of images in each category folder
        imageID: List with imageID (add this information somewhere in the stimuli initialization to document which images are used)
        
        images_in_each_category = dictionary of size 2 (if 2 categories), with key = category name, and value = list containing image paths
        
        
        ?
    """

    image_formats = {'jpg', 'jpeg', 'pgm'}

    categories = findCategories(directory)
    no_categories = len(categories)
    # class_indices = dict(zip(classes, range(num_class)))

    noImages = 0
    images_in_each_category = {}
    for subdir in categories:
        subpath = os.path.join(directory, subdir)
        for root, _, files in recursiveList(subpath):
            for fname in files:
                is_valid = False
                for extension in image_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    noImages += 1
                images_in_each_category[subdir] = [subpath + '\\' + ii for ii in files]
    print('Found %d images belonging to %d categories.\n' % (noImages, no_categories))
    return noImages, images_in_each_category

def createRandomImages(dom='woman',lure='man'):
    '''Takes two input categories, and draws 45 images from the dominant category, and 5 from the lure category
    
    # Returns
        fusedList: One list consisting of 50 images from dominant and lure categories in random order
        fusedCats: One list of the category names from fusedList (used in directory indexing)
    
    '''
    categories = findCategories(data_path) 
    noImages, images_in_each_category = findImages(data_path)
    
    for key, value in images_in_each_category.items():        
        if key == dom:
            randomDom = sample(value, 10) # Randomly takes X no of samples from the value list corresponding to that category
        if key == lure:
            randomLure = sample(value, 1)
            
    fusedList = randomDom + randomLure
    random.shuffle(fusedList)
    
    fusedCats = []
    for item in fusedList:
        str_item = str(item)
        str_item = str_item.split('\\')
        fusedCats.append(str_item[-2])
    
    return fusedList, fusedCats

def fuseStableImages(aDom, aLure, nDom, nLure): #make arguments that decide which categories to take and input in create random imgs
    """Returns a fused image with alpha 0.5
        
    # Arguments:
        directory - make default
        batch1: 50 images consisting of 45 dominant images and 5 lures
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks with 
        
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

    
    # Add the imageIDs into a .csv and divide into blocks. write to this file
    
    # delete stable_save after it has been used? 
    
    # with open(data_path + '\log_fuseStableImages.csv', 'a') as logFile:
    #    logFile.write('NEW BLOCK') 
    
#    with open(data_path + '\logs' '\log_fuseStableImages.csv', 'a') as logFile: #dont open it each time
#            logFile.write(aCatImages[i] + nCatImages[i])
#            logFile.writerow('hhh')
#            logFile.write('\n')

def fuseImages(directory, alpha):
    """Returns a fused image
    
    MAKE TWO DIFFERENT MODES: for stable blocks and NF. Alternatively, make two different fuseImages functions, and call them separately depending on which block is running
    
    This function should take an input from the processing pipeline
    
    # Arguments:
        directory
        alpha: determines the degree of background vs. foreground visibility
        
    # Returns
        ?? fused image - save, or feed directly into a different function?
    
    
    """
    categories = findCategories(data_path) 
    noImages, images_in_each_category = findImages(data_path)
    # Use the found images (function: findImages) as input to fuseImages.
    # Shuffle randomly
    
    # if nf-mode (neurofeedback mode): use parameter
    
    # if block-mode: use a set parameter for fusing images
    
    # Add imageID as data_path + imageID
    
    # Make a shuffled random list and draw imageIDs from those
    # Somehow combine e.g. indoor + outdoor
    
    # Make two modes
    
    background = Image.open(os.path.join(data_path + '\scenes' + imageID), mode='r')
    foreground = Image.open(os.path.join(data_path,'\faces' + imageID))

    fusedImage=Image.blend(background, foreground, .2)
    
    # Add the imageID and alpha value to a list?
    
    
def initialize(data_path, visualangle=[4,3]):
    duration_image = 1

    # Should initialize take an input from fuseImages, and add it to the trial list?

    images, images_in_each_category = findImages(data_path)
    trial_list = []

    items = images_in_each_category.items() # Is this correctly made into a dict in findImages??
    # random.shuffle(items) # Create a way of shuffling the images at some point

    ready_statement = {'probeword': 'ready', 'condsFile': None}
    done_statement = {'probeword': 'done! thanks', 'condsFile': None}
    break_statement = {'probeword': 'break', 'condsFile': [{'duration': duration_image}]}

    trial_list.append(ready_statement)
    count = 1
    for key, value in items:

        trial = {}
        temp_condsfile = []

        # Inserting ready and breaks. INSERT BREAKS EVERY ????
#        if count % 4 == 0:
#            trial_list.append(break_statement)
#            trial_list.append(ready_statement)

#         trial['probeword'] = key.decode('iso-8859-1')


        for v in value:
            temp_condsfile.append({'duration': duration_image,
                                   'visualangle': visualangle})

        trial['condsFile'] = temp_condsfile
        trial_list.append(trial)
        
        trial_list.append(value)

        count += 1

    trial_list.append(done_statement)
    return trial_list
    
    
############ TESTING PSYCHOPY #################

# Initializing window
win = visual.Window(
    size=[500, 500], fullscr=False, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

# Initializing fixation
fixation_text = visual.TextStim(win=win, name='fixation_text',
                                text='+',
                                font='Arial',
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1,
                                depth=-1.0)

# Initializing clock:
testClock = core.Clock()

# Initializing trial numbers
num_trials = 15 #50
num_dom = 10 #45
num_lure = 5

catDict = {'woman':'man','man':'woman','indoor':'outdoor','outdoor':'indoor',
           'woman':'man','man':'woman','indoor':'outdoor','outdoor':'indoor'} 
# 8 pairs, since I have 8 blocks for 1 run


catFacesDict = {'woman':'man','man':'woman','woman':'man','man':'woman'} 
catScenesDict = {'indoor':'outdoor','outdoor':'indoor', 'indoor':'outdoor','outdoor':'indoor'}

for pair in catFacesDict:
    print(pair)

    
catFacesLst = [['woman','man'],['man','woman'],['woman','man'],['man','woman']] 
    
for pair in catFacesLst:
    print(pair)
    
# When running a 8 stable blocks

random.shuffle(catFacesLst) #Shuffle randomly

for item in catFacesLst:
    domInput = item[0]
    lureInput = item[1]

catFacesLst.pop(0) # Delete the category pair that has just been used as input 


# Generate overall input list and simply take from this one

# if stable block... do 8

# if NF stable ...... do 4 


catComb = [['man','woman','indoor','outdoor'],
           ['man','woman','outdoor','indoor'],
           ['woman','man','indoor','outdoor'],
           ['woman','man','outdoor','indoor'],
           ['indoor','outdoor','man','woman'],
           ['indoor','outdoor','woman','man'],
           ['outdoor','indoor','man','woman'],
           ['outdoor','indoor','woman','man']]

# shuffle 

# Create inputs for fuseStableImages

stabeBlockLen = 8

# To generate the folders with images. Do not delete, but simply draw from them, and shuffle image order before
for k in range(0,8):
    aDom = catComb[k][0]
    aLure = catComb[k][1]
    nDom = catComb[k][2]
    nLure = catComb[k][3] 
    fuseStableImages(aDom, aLure, nDom, nLure)  
    
    images = [visual.ImageStim(win, image = data_path + '\stable_save\\no_%d.jpg' % stableIDs[idx_image]) for idx_image in range(len(stableIDs))] 
    
    for trial_idx in range(len(stableIDs)):
        
        for frameN in range(num_frames_trials):
        
            if 0 <= frameN < num_frames_stimuli: 
                images[trial_idx].draw()
        
            win.flip()
    
    # Now run the stable block from this function?
    
    # Now stable_save has to correct imgs 
    
    # Run the experiment and delete all imgs in stable_save 


stableIDs = list(range(0,11)) # The numbering of the images in stable_save
frame_rate = 60
stimuli_time = 1
trials = 11 # images in block # 50


num_frames_period = np.array([660, 660, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000]) #number of framer per each trial. Right now the same
num_frames_stimuli = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]) #number of framer per each trial. Right now the same

num_frames_period = num_frames_period.astype('int') # Denotes how long the trial periods are, e.g. 11 imgs x 1 s x 60 Hz = 660 frames
num_frames_stimuli = num_frames_stimuli.astype('int') # Denotes how many frames a stimuli is shown


images = [visual.ImageStim(win, image = data_path + '\stable_save\\no_%d.jpg' % stableIDs[idx_image]) for idx_image in range(len(stableIDs))] 
    

for trial in range(trials): # right now I have 11 trials
    trial_clock = core.Clock()
    
    for frameN in range(num_frames_period[trial]): # Time x frame rate
        if frameN == 0:
            print('frame0')
    
        if 0 <= frameN < num_frames_stimuli[trial]: 
            images[trial].draw()
    
            win.flip()
        
        if frameN > num_frames_stimuli[trial]:
            print('above 60hz')
            
            win.flip()


log('Experiment started')
for trial_idx in range(num_trials):
	trial_clock = core.Clock()

	log('Beginning trial %d' % trial_idx)

	for frameN in range(num_frames_period[trial_idx]):

		# play sounds
		if frameN == 0:
			open_sound.play()
			pass
		elif frameN == num_frames_open[trial_idx]:
			closed_sound.play()

		# visual
		if 0 <= frameN < num_frames_open[trial_idx]: 
			images[trial_idx].draw()

		if num_frames_open[trial_idx] <= frameN:  # present stim for a different subset
			fixation.draw()

	    # switch buffer
		win.flip()

	log('Ended trial %d' % trial_idx)

	win.clearBuffer()
	win.flip()

	trial_time = trial_clock.getTime()












######################### GUI ##########################

# Store info about the experiment session
#expName = 'image_experiment'  # from the Builder filename that created this script
############
#dlg = gui.Dlg(title=expName)
#dlg.addText('Subject info')
#dlg.addField('Name:', "")
#dlg.addField('Age:', 21)
#dlg.addField('Note:', "")
#dlg.addField('Gender:', choices=["Male", "Female"])
#dlg.addText('Experiment Info')
#dlg.addField('Folder name:', "experiment_data/expXXX")
#dlg.addField('Setup:', choices=["Test", "Execution"])
#dlg.addText('Before clicking OK remember to activate LSL', color='red')
#ok_data = dlg.show()  # show dialog and wait for OK or Cancel
#ok_data = np.asarray([str(ii) for ii in ok_data])
#if not dlg.OK:  # or if ok_data is not None
#    core.quit()  # user pressed cancel
#else:
#
#    if ok_data[5] == 'Execution':
#        exp_time = time.localtime()
#
#        trialList = initialize(data_path)
#        experiment_path = data_path + ok_data[4]
#        file = open(data_path + '/info.txt', "w")
#
#        file.write('Name ' + ok_data[0] + '\n')
#        file.write('Age ' + ok_data[1] + '\n')
#        file.write('Note ' + ok_data[2] + '\n')
#        file.write('Gender ' + ok_data[3] + '\n')
#        file.write('Date ' + str(exp_time.tm_mday) + '/' + str(exp_time.tm_mon) + '/' + str(exp_time.tm_year) + ' ' + str(exp_time.tm_hour) + ':' + str(exp_time.tm_min))
#
#        file.close()
#
#        print('Input saved')
#
#    if ok_data[5] == 'Test':
#        trialList = initialize(data_path)
