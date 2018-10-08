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


def fuseStableImages2(aDom, aLure, nDom, nLure): #make arguments that decide which categories to take and input in create random imgs
    '''
        Returns a fused image with alpha 0.5 based on a chosen attentive category, and a chosen unattentive category.
        Saves the fused images in a folder named after the attentive category.
    
    # Arguments:
        Names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks
        
        Save the images? Delete them afterwards
        Save the image IDs to a log file
    '''

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure) 
    
    # Save information about the attentive category for later probe word text display
    if aDom == 'man' or aDom == 'woman':
        aFolder = 'faces' # Attentive folder name

    if aDom == 'indoor' or aDom == 'outdoor':
        aFolder = 'scenes'

    imageCount = 0
    global stableSaveCount
    
    # Make directory based on a global variable count for saving the fused images: stableSaveCount
    newFolderPath = r'C:\\Users\\Greta\\Documents\GitHub\closed_loop\data\stable\\' + aFolder + str(stableSaveCount) + '\\' 
    if not os.path.exists(newFolderPath):
        os.makedirs(newFolderPath)
        
#    df = pd.DataFrame(columns=['img1', 'img2']) # Initializing log pd df file
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        fusedImage = Image.blend(background, foreground, .5)
    
        fusedImage.save(data_path + '\stable' + '\\' + aFolder + str(stableSaveCount) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        fusedImage.close()
    
        imageCount += 1
        
        # TESTING DF
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))] = [aCatImages[i], nCatImages[i]] # Writes to df in order to not overwrite 
        
    print('Created {0} fused images in {2}{1} with alpha 0.5.\n'.format(len(aCatImages), str(stableSaveCount), aFolder))

    df_stableSave.loc[i + stableSaveCount * len(aCatImages)] = ['NEW BLOCK']
    df_stableSave.to_csv(data_path + '\logs' '\TEST_fuseStableImages.csv')

    stableSaveCount += 1
    
    del aDom, aLure, nDom, nLure, aFolder 

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
        
    
############ PSYCHOPY #################

# Initializing window
win = visual.Window(
    size=[500, 500], fullscr=False, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

# Initializing fixation
textFix = visual.TextStim(win=win, name='textFix',
                                text='+',
                                font='Arial',
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1,
                                depth=-1.0)

# Initializing stimuli presentation times (in Hz)
frameRate = 60
stimuliTime = 60 



timeLst = []

textFaces = visual.TextStim(win=win, name='textFaces',
                                 text='faces',
                                 font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1,
                                 depth=0.0)

textScenes = visual.TextStim(win=win, name='textScenes',
                                 text='scenes',
                                 font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1,
                                 depth=0.0)

# Initializing clock:
testClock = core.Clock()

def runBlock(images,textInput):
    '''Initializes a single block with text for 1 s, fixation cross, and trials for 1 s.
    
    # Arguments
        images: from initializeStableBlock, chooses images generated from a chosen foldername
        text: which category the subject should attend to. Either scenes or faces.
    
    '''
    if textInput == 'faces':
        textProbe = textFaces
    if textInput == 'scenes':
        textProbe = textScenes

    
    for frameN in range(0,150):
        textProbe.draw()
        win.flip()
    
    # INSET fixation time here
    for frameC in range(0,300):
        textFix.draw()
        win.flip()
    
              
    imgCounter = 0 # Count of which image in "images" to take

    for frameB in range(0,660): # 660 is 60*11, but it should be 60*50
        if frameB % 60 == 0 and imgCounter <= 9:
            imgCounter += 1
            for newframe in range(0,60):
                if newframe >= 0:
                    images[imgCounter].draw()
                
                win.flip()
        


def initializeStableBlock(folderName):
    ''' Initializes the variable "images" later used for psychopy function in a folder
    
    # Arguments
        Which folder to take
    
    # Returns
        Calls runBlock, so that images are used for that.

    '''
    
    textInput = folderName[:-1] # Removing the digit from the folder, in order  to use the folder name for probe word input
    
    images = [visual.ImageStim(win, image = data_path + '\stable\\' + folderName + '\\no_%d.jpg' % trialIDs[idx_image]) for idx_image in range(len(trialIDs))] 
    runBlock(images,textInput) #Calling runBlock with the image input
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
