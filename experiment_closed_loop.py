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
global stableSaveCount 
stableSaveCount = 1 

trialIDs = list(range(0,11)) # The numbering of the images in stable_save.
# Denotes the numbering of images in folders, will always be a list of length 50


# Data frames for writing log files
df_stableSave = pd.DataFrame(columns=['img1', 'img2'])



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

def fuseStableImages2(aDom, aLure, nDom, nLure): #make arguments that decide which categories to take and input in create random imgs
    """Returns a fused image with alpha 0.5, and fuses all images in arg1 folder with all images in arg2 folder
        Would not work - need to 
        
        CREATE FOLDERS, too.
        
    # Arguments:
        batch1: 50 images consisting of 45 dominant images and 5 lures
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks with 
        
        Save the images? Delete them afterwards
        Save the image IDs to a log file
    
    """    

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure)  

    imageCount = 0
    global stableSaveCount
    
    # Make directory based on a global variable count for saving the fused images: stableSaveCount
    newFolderPath = r'C:\\Users\\Greta\\Documents\GitHub\closed_loop\data\stable_save' + str(stableSaveCount) + '\\' 
    if not os.path.exists(newFolderPath):
        os.makedirs(newFolderPath)
        
#    df = pd.DataFrame(columns=['img1', 'img2']) # Initializing log pd df file
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        fusedImage = Image.blend(background, foreground, .5)
    
        fusedImage.save(data_path + '\stable_save' + str(stableSaveCount) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        fusedImage.close()
    
        imageCount += 1
        
        # TESTING DF
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))] = [aCatImages[i], nCatImages[i]] # Writes to df in order to not overwrite 
        
    print('Created {0} fused images in stable_save{1} with alpha 0.5.\n'.format(len(aCatImages), str(stableSaveCount)))

    df_stableSave.loc[i + stableSaveCount * len(aCatImages)] = ['NEW BLOCK']
    df_stableSave.to_csv(data_path + '\logs' '\TEST_fuseStableImages.csv')

    stableSaveCount += 1

    # Add the imageIDs into a .csv and divide into blocks. write to this file
    
    # delete stable_save after it has been used? 
    
    # with open(data_path + '\log_fuseStableImages.csv', 'a') as logFile:
    #    logFile.write('NEW BLOCK') 
    
#    with open(data_path + '\logs' '\log_fuseStableImages.csv', 'a') as logFile: #dont open it each time
#            logFile.write(aCatImages[i] + nCatImages[i])
#            logFile.writerow('hhh')
#            logFile.write('\n')

def fuseImages(directory, alpha): # NOT DONE
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
fixation = visual.GratingStim(win, tex=None, mask='gauss', sf=0, size=0.02,
    name='fixation', autoLog=False)

# Initializing clock:
testClock = core.Clock()

# Initializing trial numbers
num_trials = 15 #50
num_dom = 10 #45
num_lure = 5
 
def generateStableFolders():
    '''Generates all possible combinations of images fused with alpha 0.5
    
    is currently done in the main
    
    
    '''

frame_rate = 60
stimuli_time = 1 # in seconds
trials = 11 # images in block # 50

num_frames_period = np.array([780, 780, 780, 780, 3000, 3000, 3000, 3000, 3000, 3000, 3000]) #number of framer per each trial. Right now the same
num_frames_stimuli = np.array([60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60]) #number of framer per each trial. Right now the same



num_frames_period = num_frames_period.astype('int') # Denotes how long the trial periods are, e.g. 11 imgs x 1 s x 60 Hz = 660 frames
num_frames_stimuli = num_frames_stimuli.astype('int') # Denotes how many frames a stimuli is shown

timeLst = []

probeword_text = visual.TextStim(win=win, name='probeword_text',
                                 text='yooooo',
                                 font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1,
                                 depth=0.0)


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
        
#            if frameN <= num_frames_stimuli[trial]:  
#                fixation_text.draw()
#                
#                win.flip()
            
#            if frameN > num_frames_stimuli[trial]:
#                print('above 60hz')
#                
#                win.flip()


def initializeStableBlock(folderName):
    ''' Initializes the variable "images" later used for psychopy function in a folder
    
    # Arguments
        Which folder to take
    
    # Returns
        Make it call the show image function, so that images are used for that.

    '''
    
    images = [visual.ImageStim(win, image = data_path + '\stable\\' + folderName + '\\no_%d.jpg' % trialIDs[idx_image]) for idx_image in range(len(trialIDs))] 
    runBlock(images) #Calling runBlock with the image input
    print('In the initialize stable blocks loop')

#    
#if event.getKeys(keyList=["escape"]):
#    win.close()


#
#log('Experiment started')
#for trial_idx in range(num_trials):
#	trial_clock = core.Clock()
#
#	log('Beginning trial %d' % trial_idx)
#
#	for frameN in range(num_frames_period[trial_idx]):
#
#		# play sounds
#		if frameN == 0:
#			open_sound.play()
#			pass
#		elif frameN == num_frames_open[trial_idx]:
#			closed_sound.play()
#
#		# visual
#		if 0 <= frameN < num_frames_open[trial_idx]: 
#			images[trial_idx].draw()
#
#		if num_frames_open[trial_idx] <= frameN:  # present stim for a different subset
#			fixation.draw()
#
#	    # switch buffer
#		win.flip()
#
#	log('Ended trial %d' % trial_idx)
#
#	win.clearBuffer()
#	win.flip()
#
#	trial_time = trial_clock.getTime()












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
