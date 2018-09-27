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
import sys  
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from psychopy import gui, visual, core, data, event, monitors
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
import time 
import numpy as np
import matplotlib.pyplot as plt
from settings import path_init


data_path = path_init()

# jpgfile = Image.open(data_path + "th.jpg")

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
                images_in_each_category[subdir] = [subpath + '/' + ii for ii in files]
    print('Found %d images belonging to %d categories.\n' % (noImages, no_categories))
    return noImages, images_in_each_category






def fuseImages(directory, alpha):
    """Returns a fused image
    
    MAKE TWO DIFFERENT MODES: for stable blocks and NF. Altnernatively, make two different fuseImages functions, and call them separately depending on which block is running
    
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
    
    background = Image.open(os.path.join(data_path + '/scenes/' + imageID), mode='r')
    foreground = Image.open(os.path.join(data_path,'/faces/' + imageID))

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

# Following works:
message = visual.TextStim(win, text='hello')
message.setAutoDraw(True)  # automatically draw every frame
win.flip()
core.wait(2.0)
message.setText('world')  # change properties of existing stim
win.flip()
core.wait(2.0)

# Initializing clock:
testClock = core.Clock()

while testClock.getTime()<0:





#
## Initialize components for Routine "probe_word"
#probe_wordClock = core.Clock()
#
#probeword_text = visual.TextStim(win=win, name='probeword_text',
#                                 text='default text',
#                                 font='Arial',
#                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
#                                 color='white', colorSpace='rgb', opacity=1,
#                                 depth=0.0)
#
## Initialize components for Routine "trial"
#trialClock = core.Clock()
#image = visual.ImageStim(
#    win=win, name='image', units='deg',
#    image='sin', mask=None,
#    ori=0, pos=(0, 0), size=1.0,
#    color=[1, 1, 1], colorSpace='rgb', opacity=1,
#    flipHoriz=False, flipVert=False,
#    texRes=128, interpolate=True, depth=0.0)
#
#fixation_text = visual.TextStim(win=win, name='fixation_text',
#                                text='+',
#                                font='Arial',
#                                pos=(0, 0), wrapWidth=None, ori=0,
#                                color='white', colorSpace='rgb', opacity=1,
#                                depth=-1.0)




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
#
#expInfo = {u'session': u'001', u'participant': u''}
#
#expInfo['date'] = data.getDateStr()  # add a simple timestamp
#expInfo['expName'] = expName
#
#endExpNow = False  # flag for 'escape' or other condition => quit the exp
#
#win = visual.Window(
#    size=[3840, 2160], fullscr=True, screen=0,
#    allowGUI=False, allowStencil=False,
#    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
#    blendMode='avg', useFBO=True)
#
## store frame rate of monitor if we can measure it
#expInfo['frameRate'] = win.getActualFrameRate()
#if expInfo['frameRate'] != None:
#    frameDur = 1.0 / round(expInfo['frameRate'])
#else:
#    frameDur = 1.0 / 60.0  # could not measure, so guess
#
## Initialize components for Routine "probe_word"
#probe_wordClock = core.Clock()
#
#probeword_text = visual.TextStim(win=win, name='probeword_text',
#                                 text='default text',
#                                 font='Arial',
#                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
#                                 color='white', colorSpace='rgb', opacity=1,
#                                 depth=0.0)
#
## Initialize components for Routine "trial"
#trialClock = core.Clock()
#image = visual.ImageStim(
#    win=win, name='image', units='deg',
#    image='sin', mask=None,
#    ori=0, pos=(0, 0), size=1.0,
#    color=[1, 1, 1], colorSpace='rgb', opacity=1,
#    flipHoriz=False, flipVert=False,
#    texRes=128, interpolate=True, depth=0.0)
#
#fixation_text = visual.TextStim(win=win, name='fixation_text',
#                                text='+',
#                                font='Arial',
#                                pos=(0, 0), wrapWidth=None, ori=0,
#                                color='white', colorSpace='rgb', opacity=1,
#                                depth=-1.0)
#
#
#globalClock = core.Clock()  # to track the time since experiment started
#routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 
#
#while continueRoutine and routineTimer.getTime() > 0:
#        # get current time
#        t_probe = probe_wordClock.getTime()
#        frameN += 1  # number of completed frames
#        # update/draw components on each frame
#
#        # *probeword_text* updates
#        if t_probe >= 0.0 and probeword_text.status == NOT_STARTED:
#            # keep track of start time/frame for later
#            probeword_text.tStart = t_probe
#            probeword_text.frameNStart = frameN  # exact frame index
#            probeword_text.setAutoDraw(True)
#            first_probe_word = True
#        if probeword_text.status == STARTED and t_probe >= frameRemains:
#            probeword_text.setAutoDraw(False)
#            probe_word_2_lsl = False
#
#        # check if all components have finished
#        if not continueRoutine:  # a component has requested a forced-end of Routine
#            break
#
#        continueRoutine = False  # will revert to True if at least one component still running
#        for thisComponent in probe_wordComponents:
#            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
#                continueRoutine = True
#                break  # at least one component has not yet finished
#
#        # refresh the screen
#        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
#            if probeword_text.autoDraw and first_probe_word:
#                first_probe_word = False
#
#                if make_ready_lsl_stream:
#                    outlet.push_sample(['ready'])
#                    make_ready_lsl_stream = False
#
#                if image_2_lsl:
#                    image_2_lsl = False
#                    outlet.push_sample([imagetest])
#
#                if first_long_break:
#                    first_long_break = False
#                    outlet.push_sample(['long break'])
#
#                if probeword == 'ready':
#                    make_ready_lsl_stream = True
#
#                win.flip()
#                #######################################
#                outlet.push_sample([probeword])
#                #######################################
#
#        # check for quit (the Esc key)
#        if endExpNow or event.getKeys(keyList=["escape"]):
#            core.quit()

