# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:08:54 2018

@author: gretatuckute
"""
# Imports
import os
os.chdir('C:\\Users\\Greta\\Documents\GitHub\closed_loop')

from PIL import Image
import os 
import random
from random import sample
import sys  
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_stream
import numpy as np
from psychopy import gui, visual, core, data, event, monitors, logging
#from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
import time 
import numpy as np
from paths import data_path_init, log_path_init, stable_path_init
import csv
import pandas as pd

data_path = data_path_init()
log_path = log_path_init()
stable_path = stable_path_init()

############### Global variables ###############

global stableSaveCount 
stableSaveCount = 1 
global streams 

global imgIdx
imgIdx = 1

############### Data frames for logging ###############
df_stableSave = pd.DataFrame(columns=['attentive cat','img1', 'img2'])


############### EXPERIMENT FUNCTIONS ###############

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
    """Returns the number of images and categories (subcategories) in a given folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        noCategories: Total number of folders/categories in the given folder - currently not dierctly returned
        noImages: Total number of images in each category folder
        imagesInEachCategory = dictionary of size 2 (if 2 categories), with key = category name, and value = list containing image paths
    """

    imageFormats = {'jpg', 'jpeg', 'pgm'}

    categories = findCategories(directory)
    noCategories = len(categories)
    # class_indices = dict(zip(classes, range(num_class)))

    noImages = 0
    imagesInEachCategory = {}
    for subdir in categories:
        subpath = os.path.join(directory, subdir)
        for root, _, files in recursiveList(subpath):
            for fname in files:
                is_valid = False
                for extension in imageFormats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    noImages += 1
                imagesInEachCategory[subdir] = [subpath + '\\' + ii for ii in files]
    print('Found %d images belonging to %d categories.\n' % (noImages, noCategories))
    return noImages, imagesInEachCategory

############### PSYCHOPY ###############

# Initializing window
win = visual.Window(
    size=[500, 500], fullscr=False, screen=0,
    allowGUI=False, allowStencil=False, autolog=True,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

# Initializing fixation text and probe word text 
textFix = visual.TextStim(win=win, name='textFix', text='+', font='Arial',
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1, depth=-1.0)

textFaces = visual.TextStim(win=win, name='textFaces', text='faces', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

textScenes = visual.TextStim(win=win, name='textScenes', text='scenes', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

# Initializing stimuli presentation times (in Hz)
frameRate = 60
probeTime = 120
fixTime = 120
stimTime = 60 # stimuli time for presenting each image
stimTimeTotal = 660
numImages = 11 # Correct: 50
trialIDs = list(range(0,11)) # The numbering of the images in stable_save. Denotes the numbering of images in folders, will always be a list of length 50


# Prepare log
log_base = time.strftime('%m-%d-%y_%H-%M-%S')
logWritePath = log_path + '\\log_' + str(log_base) + '.csv'
logWritePathKey = log_path + '\\log_key_' + str(log_base) + '.csv'

globalClock = core.Clock()
# globalClock.reset() 
logging.LogFile(logWritePath, level=logging.EXP, filemode='w')
logging.setDefaultClock(globalClock)
logging.console = True

logging.LogFile(logWritePathKey, level=logging.DATA, filemode='w') # Log file for button press only
logging.setDefaultClock(globalClock)
logging.console = True

def log(msg):
    """ For printing messages in the promt
    """
    
    logging.log(level=logging.EXP, msg=msg) 


# Initializing clock:
#trialClock = core.Clock()
#timeList = []



def createRandomImages(dom='female',lure='male'):
    """Takes two input categories, and draws 45 images from the dominant category, and 5 from the lure category
    
    # Returns
        fusedList: One list consisting of 50 images from dominant and lure categories in random order
        fusedCats: One list of the category names from fusedList (used in directory indexing)
    """
    
    categories = findCategories(data_path) 
    noImages, imagesInEachCategory = findImages(data_path)
    
    for key, value in imagesInEachCategory.items():        
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
    """ Returns a fused image with alpha 0.5 based on a chosen attentive category, and a chosen unattentive category.
        Saves the fused images in a folder named after the attentive category.
    
    # Arguments:
        Names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks
        imageID: List with imageID (add this information somewhere in the stimuli initialization to document which images are used)
        
        Save the images? Delete them afterwards
        Save the image IDs to a log file
    """

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure) 
    
    # Save information about the attentive category for later probe word text display
    if aDom == 'male' or aDom == 'female':
        aFolder = 'faces' # Attentive folder name

    if aDom == 'indoor' or aDom == 'outdoor':
        aFolder = 'scenes'

    imageCount = 0
    global stableSaveCount
    
    # Make directory based on a global variable count for saving the fused images: stableSaveCount
    # newFolderPath = r'C:\\Users\\Greta\\Documents\GitHub\closed_loop\data\stable\\' + aFolder + str(stableSaveCount) + '\\' 
    newFolderPath = stable_path + '\\' + aFolder + str(stableSaveCount) + '\\' 
    if not os.path.exists(newFolderPath):
        os.makedirs(newFolderPath)
        
#    df = pd.DataFrame(columns=['img1', 'img2']) # Initializing log pd df file
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        fusedImage = Image.blend(background, foreground, .5)
    
        fusedImage.save(stable_path + '\\' + aFolder + str(stableSaveCount) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        fusedImage.close()
    
        imageCount += 1
        
        # Logging the image paths/IDs to a df
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))] = [aCatImages[i], nCatImages[i]] # Writes to df in order to not overwrite 
        
    print('Created {0} fused images in {2}{1} with alpha 0.5.\n'.format(len(aCatImages), str(stableSaveCount), aFolder))

    df_stableSave.loc[i + stableSaveCount * len(aCatImages)] = ['NEW BLOCK']
    df_stableSave.to_csv(log_path + '\\fuseStableImages.csv') # Add a date string or numbering to this csv file

    stableSaveCount += 1
    
    del aDom, aLure, nDom, nLure, aFolder 
    
def fuseStableImages2(aDom, aLure, nDom, nLure): #make arguments that decide which categories to take and input in create random imgs
    """ Returns a fused image with alpha 0.5 based on a chosen attentive category, and a chosen unattentive category.
        Saves the fused images in a folder named after the attentive category.
    
    # Arguments:
        Names of attentive dominant category, attentive lure category, nonattentive dominant category and nonattentive lure category.
        
    # Returns
        List of 50 fused images to use as stimuli in stable blocks
        imageID: List with imageID (add this information somewhere in the stimuli initialization to document which images are used)
        
        Save the images? Delete them afterwards
        Save the image IDs to a log file
    """

    aCatImages, aCats = createRandomImages(dom=aDom,lure=aLure) # 50 attend category images
    nCatImages, nCats = createRandomImages(dom=nDom,lure=nLure) 
    
    # Save information about the attentive category for later probe word text display
    if aDom == 'male' or aDom == 'female':
        aFolder = 'faces' # Attentive folder name

    if aDom == 'indoor' or aDom == 'outdoor':
        aFolder = 'scenes'

    imageCount = 0
    global stableSaveCount
    
    for i in range(len(aCatImages)):
        
        background = Image.open(os.path.join(aCatImages[i]), mode='r')
        foreground = Image.open(os.path.join(nCatImages[i]))
        
        fusedImage = Image.blend(background, foreground, .5)
    
        # fusedImage.save(stable_path + '\\' + aFolder + str(stableSaveCount) + '\\' 'no_' + str(imageCount) + '.jpg')
    
        background.close()
        foreground.close()
        fusedImage.close()
    
        imageCount += 1
        
        # Logging the image paths/IDs to a df
        df_stableSave.loc[i + (stableSaveCount * len(aCatImages))] = [aDom, aCatImages[i], nCatImages[i]] # Writes to df in order to not overwrite 
        
    print('Created {0} fused images in {2}{1} with alpha 0.5.\n'.format(len(aCatImages), str(stableSaveCount), aFolder))

    # df_stableSave.loc[i + stableSaveCount * len(aCatImages)] = ['NEW BLOCK']
    df_stableSave.to_csv(log_path + '\\fuseStableImages2.csv') # Add a date string or numbering to this csv file

    stableSaveCount += 1
    
    del aDom, aLure, nDom, nLure, aFolder 
    
def fuseImgs(csvfile):
    """Currently calls runImg
    
    Outputs a single fused img based on csv index
    
    
    """
    # Read from csv file here and use the img IDs as input
    global imgIdx
    
    with open(csvfile) as csv_file:
    #with open(log_path + '\\fuseStableImages.csv') as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
    
    # Implement some kind of global count that knows which row to take
    
        row_I_want = imgIdx # counter...
        
        rowInfo = csv_reader[row_I_want]
        print(rowInfo)
        
        foregroundID = rowInfo[2]
        backgroundID = rowInfo[3]
        
    imgIdx += 1
    
    background = Image.open(foregroundID, mode='r')
    foreground = Image.open(backgroundID, mode='r')
    
    fusedImg = Image.blend(background, foreground, .5)
    
    # fusedImage.show()

    runImg(fusedImg)
    
    background.close()
    foreground.close()

def runImg(fusedImg):
    
    image = visual.ImageStim(win, autoLog = True, image = fusedImg)
    
    for frameNew in range(0,60): # 0,stimTime
        if frameNew >= 0:
            image.draw()
        win.flip()

textFix = visual.TextStim(win=win, name='textFix', text='+', font='Arial',
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1, depth=-1.0)

textFemale = visual.TextStim(win=win, name='textFemale', text='female', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

textMale = visual.TextStim(win=win, name='textMale', text='male', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

textIndoor = visual.TextStim(win=win, name='textIndoor', text='indoor', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)

textOutdoor = visual.TextStim(win=win, name='textOutdoor', text='outdoor', font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)


def runProbeFix(csvfile):
    with open(csvfile) as csv_file:
        csv_reader = list(csv.reader(csv_file, delimiter=','))
        
        row_I_want = imgIdx
        
        rowInfo = csv_reader[row_I_want]
        print(rowInfo)
        
        attentiveText = str(rowInfo[1])
        # make into a string
        
        textGeneral = visual.TextStim(win=win, name='textGeneral', text=attentiveText, font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1, depth=0.0)    
    
#    for frameN in range(0,probeTime):
#        textGeneral.draw()
#        win.flip()
    for frameN in range(0,probeTime):
        textGeneral.draw()
        win.flip()
    
    for frameN in range(0,fixTime):
        textFix.draw()
        win.flip()

# testing whether fuseImgs and runImg works
        
for ii in list(range(0,20)):
    

    if ii % 10 == 0:
        runProbeFix(log_path + '\\fuseStableImages2.csv')
    fuseImgs(log_path + '\\fuseStableImages2.csv')

    

def runBlock(images,textInput):
    """Initializes a single block with text for 1 s, fixation cross, and trials for 1 s.
    
    # Arguments
        images: from initializeStableBlock, chooses images generated from a chosen foldername
        text: which category the subject should attend to. Either scenes or faces.
    """
    
#    trialClock.reset()
#    t = trialClock.getTime()
#    timeList.append(t)
    log('Running a stable block')
    
    if textInput == 'faces':
        textProbe = textFaces
    if textInput == 'scenes':
        textProbe = textScenes

    for frameN in range(0,probeTime):
        textProbe.draw()
        win.flip()
    
    for frameN in range(0,fixTime):
        textFix.draw()
        win.flip()
              
    imgCounter = 0 # Count of which image in "images" to take, used for indexing "images"

    for frameN in range(0,11): 
        #if imgCounter <= 10: 
        for frameNew in range(0,stimTime):
            if frameNew >= 0:
                images[imgCounter].draw()
            win.flip()
            
        imgCounter += 1
            

def initializeStableBlock(folderName):
    """Initializes the variable "images" later used for psychopy function in a folder
    
    # Arguments
        Which folder to take
    
    # Returns
        Calls runBlock, so that images are used for display 
    """
    
    textInput = folderName[:-1] # Removing the digit from the folder, in order  to use the folder name for probe word input
    
    images = [visual.ImageStim(win, autoLog = True, image = stable_path + '\\' + folderName + '\\no_%d.jpg' % trialIDs[idx_image]) for idx_image in range(len(trialIDs))] 
    runBlock(images,textInput) #Calling runBlock with the image input
    print('In the initialize stable blocks loop')
    
def initializeStableBlock2(csvname):
    """Initializes the variable "images" later used for psychopy function in a folder
    
    # Arguments
        csv log file with img IDs to use
    
    # Returns
        Calls runBlock, so that images are used for display 
    """
    
    textInput = folderName[:-1] # Removing the digit from the folder, in order  to use the folder name for probe word input
    
    images = [visual.ImageStim(win, autoLog = True, image = stable_path + '\\' + folderName + '\\no_%d.jpg' % trialIDs[idx_image]) for idx_image in range(len(trialIDs))] 
    runBlock(images,textInput) #Calling runBlock with the image input
    print('In the initialize stable blocks loop')
    
def closeWin():
    win.close()


# Initializing button press
keys = event.getKeys(keyList=None, timeStamped=globalClock)

if event.getKeys(keyList=["escape"]):
    win.close()

#################### EEG FUNCTIONS ###################
    
def findStream():    
    '''Searches for an EEG stream and  '''
    global streams

    streams = resolve_stream('type', 'EEG') # Searches for EEG stream
    # Make streams global??
    
    if len(streams) != 0:
        return True
    
    return False
    
    
def pullEEG():
    inlet = StreamInlet(streams[0])
    eeg_data, timestamp = inlet.pull_chunk(timeout=2.0, max_samples=32)
#    eeg = np.array(eeg_data)
    
    return data
    

def chunkSizeCorrect():
    ''' '''

def epochEEG(data, samples_epoch, samples_overlap=0):
    """Extract epochs from an EEG time series.

    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]

    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples

    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs =  int(np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs

def computeClassificationVector(eegdata, fs):
    '''Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    '''

def computeClassification(epochs, fs):
    """
    Call computeClassificationVector for each EEG epoch - NOT DETERMINED YET 
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = computeClassificationVector(epochs[:, :, i_epoch], fs).T
            feature_matrix = np.zeros((n_epochs, feat.shape[0])) # Initialize feature_matrix

        feature_matrix[i_epoch, :] = computeClassificationVector(
                epochs[:, :, i_epoch], fs).T

    return feature_matrix

def trainClassifier(feature_matrix_0, feature_matrix_1):
    """Train a binary classifier based on the stable blocks.

    First perform Z-score normalization, then fit?

    Args:
        feature_matrix_0 (numpy.ndarray): array of shape (n_samples,
            n_features) with examples for Class 0
        feature_matrix_1 (numpy.ndarray): array of shape (n_samples,
            n_features) with examples for Class 1
        
    Returns:
        (sklearn object): trained classifier (scikit object)
        (numpy.ndarray): normalization mean
        (numpy.ndarray): normalization standard deviation
    """
    # Create vector Y (class labels)
    class0 = np.zeros((feature_matrix_0.shape[0], 1))
    class1 = np.ones((feature_matrix_1.shape[0], 1))

    # Concatenate feature matrices and their respective labels
    y = np.concatenate((class0, class1), axis=0)
    features_all = np.concatenate((feature_matrix_0, feature_matrix_1),
                                  axis=0)

    # Normalize features columnwise
    mu_ft = np.mean(features_all, axis=0)
    std_ft = np.std(features_all, axis=0)

    X = (features_all - mu_ft) / std_ft

    # Train SVM using default parameters
    clf = svm.SVC()
    clf.fit(X, y)
    score = clf.score(X, y.ravel())

    return clf, mu_ft, std_ft, score



######################### GUI ##########################

# Store info about the experiment session
