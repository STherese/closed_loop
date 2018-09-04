# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 08:26:25 2018

@author: sofha
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.85.3),
    on December 15, 2017, at 13:47
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division, absolute_import, print_function

import os  # handy system and path functions
import random
import sys  # to get file system encoding
from pylsl import StreamInfo, StreamOutlet
import numpy as np
from psychopy import gui, visual, core, data, event, monitors
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
from mrcode.settings import path_init
from mrcode.xdf import load_xdf
import time

info = StreamInfo('PsychopyExperiment', 'Markers', 1, 0, 'string', 'myuidw43536')

# next make an outlet
outlet = StreamOutlet(info)


data_path = path_init()


def recursiveList(directory):
    """Returns list of tuples. Each tuple contains the path to the folder 
    along with the names of the files in that folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        samples: List of tuples. Each tuple contains the path to the folder 
        along with the names of the files in that folder.
    
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/NewImageClasses/"
        train_data_dir = dataPath + 'train'
        hej = sorted(os.walk(train_data_dir, followlinks=False), key=lambda tpl: tpl[0])
    """
    follow_links = False
    return sorted(os.walk(directory, followlinks=follow_links), key=lambda tpl: tpl[0])


def folderFinder(directory):
    """Finds names of folders in a given directory
    # Arguments
        directory: Path to the data.
    # Returns
        classes: List of classes in the folder.
    # Example
        directory = dataPath + "Classes"
        classes = GeneralTools.folderFinder(directory)
    """
    found_folders = []
    for subdir in sorted(os.listdir(directory)):
        if os.path.isdir(os.path.join(directory, subdir)):
            found_folders.append(subdir)
    return found_folders


def noClassesAndImages(directory):
    """Returns the number of samples in and classes in a given folder.
    
    # Arguments
        directory: Path to the data.
        
    # Returns
        samples: Total number of samples in all folders
        class_indices: Dict with integer ID and wordnet ID
        samplesInEachClass: Dict with wordnet ID and number of images
    
    # Example
        dataPath = "C:/Users/nicol/Desktop/Master/Data/NewImageClasses/"
        train_data_dir = dataPath + 'train'
        nb_train_samples, trainingSamplesInEachClass, class_indices = 
                                ImageTools.NoClassesAndImages(train_data_dir)
    """

    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

    classes = folderFinder(directory)
    num_class = len(classes)
    class_indices = dict(zip(classes, range(num_class)))

    samples = 0
    samples_in_each_class = {}
    for subdir in classes:
        subpath = os.path.join(directory, subdir)
        for root, _, files in recursiveList(subpath):
            for fname in files:
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    samples += 1
                samples_in_each_class[subdir] = [subpath + '/' + ii for ii in files]
    print('Found %d images belonging to %d classes.\n' % (samples, num_class))
    return samples, samples_in_each_class, class_indices


def initialize(image_path, visualangle=[4,3]):
    duration_image = 1

    samples, samples_in_each_class, class_indices = noClassesAndImages(image_path)
    trial_list = []

    items = samples_in_each_class.items()
    random.shuffle(items)

    ready_statement = {'probeword': 'ready', 'condsFile': None}
    done_statement = {'probeword': 'done! thanks', 'condsFile': None}
    break_statement = {'probeword': 'break', 'condsFile': [{'duration': duration_image}]}

    trial_list.append(ready_statement)
    count = 1
    for key, value in items:

        trial = {}
        temp_condsfile = []

        # Inserting ready and breaks
        if count % 4 == 0:
            trial_list.append(break_statement)
            trial_list.append(ready_statement)

        trial['probeword'] = key.decode('iso-8859-1')


        for v in value:
            temp_condsfile.append({'duration': duration_image,
                                   'visualangle': visualangle,
                                   'imagetest': v.decode('iso-8859-1')})

        trial['condsFile'] = temp_condsfile
        trial_list.append(trial)

        count += 1

    trial_list.append(done_statement)
    return trial_list


# Set the duration of the primer break
primer_break = 1.65
probeword_duration = 5
long_break = 25
random_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = 'image_experiment'  # from the Builder filename that created this script
###########
dlg = gui.Dlg(title=expName)
dlg.addText('Subject info')
dlg.addField('Name:', "")
dlg.addField('Age:', 21)
dlg.addField('Note:', "")
dlg.addField('Gender:', choices=["Male", "Female"])
dlg.addText('Experiment Info')
dlg.addField('Folder name:', "experiment_data/expXXX")
dlg.addField('Setup:', choices=["Test", "Execution"])
dlg.addText('Before clicking OK remember to activate LSL', color='red')
ok_data = dlg.show()  # show dialog and wait for OK or Cancel
ok_data = np.asarray([str(ii) for ii in ok_data])
if not dlg.OK:  # or if ok_data is not None
    core.quit()  # user pressed cancel
else:

    if ok_data[5] == 'Execution':
        exp_time = time.localtime()

        trialList = initialize(data_path + '/images')
        experiment_path = data_path + ok_data[4]
        file = open(experiment_path + '/info.txt', "w")

        file.write('Name ' + ok_data[0] + '\n')
        file.write('Age ' + ok_data[1] + '\n')
        file.write('Note ' + ok_data[2] + '\n')
        file.write('Gender ' + ok_data[3] + '\n')
        file.write('Date ' + str(exp_time.tm_mday) + '/' + str(exp_time.tm_mon) + '/' + str(exp_time.tm_year) + ' ' + str(exp_time.tm_hour) + ':' + str(exp_time.tm_min))

        file.close()

        print('Input saved')

    if ok_data[5] == 'Test':
        trialList = initialize(data_path + '/trainimages')

expInfo = {u'session': u'001', u'participant': u''}

expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# If fullscr = False we get the frameRate
# Setup the Window
# win = visual.Window(
#    size=[2560, 1440], fullscr=True, screen=1,
#    allowGUI=False, allowStencil=False,
#    monitor='screenOffice', color=[0,0,0], colorSpace='rgb',
#    blendMode='avg', useFBO=True)

# Note that it is important to choose the right monitor and that the dimensions
# of the monitor is correct in the Monitor Center. 
win = visual.Window(
    size=[3840, 2160], fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0, 0, 0], colorSpace='rgb',
    blendMode='avg', useFBO=True)

# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialize components for Routine "probe_word"
probe_wordClock = core.Clock()

probeword_text = visual.TextStim(win=win, name='probeword_text',
                                 text='default text',
                                 font='Arial',
                                 units='norm', pos=(0, 0), wrapWidth=None, ori=0,
                                 color='white', colorSpace='rgb', opacity=1,
                                 depth=0.0)

# Initialize components for Routine "trial"
trialClock = core.Clock()
image = visual.ImageStim(
    win=win, name='image', units='deg',
    image='sin', mask=None,
    ori=0, pos=(0, 0), size=1.0,
    color=[1, 1, 1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=0.0)

fixation_text = visual.TextStim(win=win, name='fixation_text',
                                text='+',
                                font='Arial',
                                pos=(0, 0), wrapWidth=None, ori=0,
                                color='white', colorSpace='rgb', opacity=1,
                                depth=-1.0)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# Change method to random to shuffle between classes or sequential to follow structure of folder
trial_structure = data.TrialHandler(nReps=1, method='sequential',
                                    extraInfo=expInfo, originPath=-1,
                                    trialList=trialList,
                                    seed=None, name='trial_structure')

this_trial_structure = trial_structure.trialList[0]  # so we can initialise stimuli with some values
image_2_lsl = False
first_long_break = False
for this_trial_structure in trial_structure:

    # abbreviate parameter names if possible (e.g. rgb = this_trial_structure.rgb)
    if this_trial_structure != None:
        for paramName in this_trial_structure.keys():
            exec (paramName + '= this_trial_structure.' + paramName)

    # ------Prepare to start Routine "probe_word"-------
    # update component parameters for each repeat
    probeword_text.setText(probeword)
    # keep track of which components have finished
    probe_wordComponents = [probeword_text]
    for thisComponent in probe_wordComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    t_probe = 0
    frameN = -1
    continueRoutine = True
    probe_wordClock.reset()  # clock
    routineTimer.reset()
    routineTimer.add(probeword_duration)
    frameRemains = probeword_duration - win.monitorFramePeriod * 0.75  # most of one frame period left

    if probeword == 'ready':
        make_ready_lsl_stream = False

    # -------Start Routine "probe_word"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t_probe = probe_wordClock.getTime()
        frameN += 1  # number of completed frames
        # update/draw components on each frame

        # *probeword_text* updates
        if t_probe >= 0.0 and probeword_text.status == NOT_STARTED:
            # keep track of start time/frame for later
            probeword_text.tStart = t_probe
            probeword_text.frameNStart = frameN  # exact frame index
            probeword_text.setAutoDraw(True)
            first_probe_word = True
        if probeword_text.status == STARTED and t_probe >= frameRemains:
            probeword_text.setAutoDraw(False)
            probe_word_2_lsl = False

        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break

        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in probe_wordComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished

        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            if probeword_text.autoDraw and first_probe_word:
                first_probe_word = False

                if make_ready_lsl_stream:
                    outlet.push_sample(['ready'])
                    make_ready_lsl_stream = False

                if image_2_lsl:
                    image_2_lsl = False
                    outlet.push_sample([imagetest])

                if first_long_break:
                    first_long_break = False
                    outlet.push_sample(['long break'])

                if probeword == 'ready':
                    make_ready_lsl_stream = True

                win.flip()
                #######################################
                outlet.push_sample([probeword])
                #######################################

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()


    if condsFile:
        # set up handler to look after randomisation of conditions etc
        # Change method to random to shuffle between images or sequential to follow structure of folder
        trials = data.TrialHandler(nReps=1, method='random',
                                   extraInfo=expInfo, originPath=-1,
                                   trialList=condsFile,
                                   seed=None, name='trials')
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values

        if probeword == 'break':
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial.keys():
                    exec (paramName + '= thisTrial.' + paramName)

            # ------Prepare to start Routine "trial"-------
            t = 0
            trialClock.reset()  # clock
            frameN1 = -1
            continueRoutine1 = True

            # keep track of which components have finished
            trialComponents = [fixation_text]
            for thisComponent in trialComponents:
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED

            frameRemainsText = long_break - win.monitorFramePeriod * 0.75  # most of one frame period left
            # -------Start Routine "trial"-------
            while continueRoutine1:
                # get current time
                t = trialClock.getTime()
                frameN1 += 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame

                # *fixation_text* updates
                if t >= 0.0 and fixation_text.status == NOT_STARTED:
                    # keep track of start time/frame for later
                    fixation_text.tStart = t
                    fixation_text.frameNStart = frameN1  # exact frame index
                    fixation_text.setAutoDraw(True)
                    first_long_break = True
                if fixation_text.status == STARTED and t >= frameRemainsText:
                    fixation_text.setAutoDraw(False)
                    first_long_break = True

                continueRoutine1 = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine1 = True
                        break  # at least one component has not yet finished

                # check if all components have finished
                if not continueRoutine1:  # a component has requested a forced-end of Routine
                    break

                # check for quit (the Esc key)
                if endExpNow or event.getKeys(keyList=["escape"]):
                    core.quit()

                # refresh the screen
                if continueRoutine1:  # don't flip if this routine is over or we'll get a blank screen
                    if fixation_text.autoDraw and first_long_break:
                        first_long_break = False
                        outlet.push_sample(['break'])
                        win.flip()
                        #######################################
                        outlet.push_sample(['long break'])
                        #######################################

        else:

            for thisTrial in trials:

                # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
                if thisTrial != None:
                    for paramName in thisTrial.keys():
                        exec (paramName + '= thisTrial.' + paramName)

                # ------Prepare to start Routine "trial"-------
                t = 0
                frameN2 = -1
                continueRoutine2 = True
                # update component parameters for each repeat
                image.setImage(imagetest)
                image.setSize(visualangle)
                image.setAutoDraw(False)
                # keep track of which components have finished
                trialComponents = [image, fixation_text]
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, 'status'):
                        thisComponent.status = NOT_STARTED

                random_interStimuli = random.choice(random_list)

                frameRemainsImage = primer_break + duration + random_interStimuli - win.monitorFramePeriod * 0.75  # most of one frame period left
                frameRemainsText = primer_break + random_interStimuli - win.monitorFramePeriod * 0.75  # most of one frame period left

                trialClock.reset()  # clock
                # -------Start Routine "trial"-------
                while continueRoutine2:
                    # get current time
                    t = trialClock.getTime()
                    frameN2 += 1  # number of completed frames (so 0 is the first frame)
                    # update/draw components on each frame

                    # *image* updates
                    if t >= frameRemainsText and image.status == NOT_STARTED:
                        # keep track of start time/frame for later
                        image.tStart = t
                        image.frameNStart = frameN2  # exact frame index
                        image.setAutoDraw(True)
                        first_image = True
                    if image.status == STARTED and t >= frameRemainsImage:
                        image.setAutoDraw(False)

                    # *fixation_text* updates
                    if t >= 0.0 and fixation_text.status == NOT_STARTED:
                        # keep track of start time/frame for later
                        fixation_text.tStart = t
                        fixation_text.frameNStart = frameN2  # exact frame index
                        fixation_text.setAutoDraw(True)
                        first_text = True
                    if fixation_text.status == STARTED and t >= frameRemainsText:
                        fixation_text.setAutoDraw(False)

                    continueRoutine2 = False  # will revert to True if at least one component still running
                    for thisComponent in trialComponents:
                        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                            continueRoutine2 = True
                            break  # at least one component has not yet finished

                    # check if all components have finished
                    if not continueRoutine2:  # a component has requested a forced-end of Routine
                        break

                    # refresh the screen
                    if continueRoutine2:  # don't flip if this routine is over or we'll get a blank screen
                        if fixation_text.autoDraw and not image.autoDraw and first_text:
                            first_text = False

                            if not probe_word_2_lsl:
                                probe_word_2_lsl = True
                                outlet.push_sample([probeword])
                                image_2_lsl = False

                            if image_2_lsl:
                                image_2_lsl = False
                                outlet.push_sample([last_imagetest])

                            win.flip()
                            #######################################
                            outlet.push_sample(['pause'])
                            #######################################
                        if image.autoDraw and first_image:
                            first_image = False
                            outlet.push_sample(['pause'])
                            win.flip()
                            #######################################
                            outlet.push_sample([imagetest])
                            #######################################
                            image_2_lsl = True
                            last_imagetest = imagetest

                    # check for quit (the Esc key)
                    if endExpNow or event.getKeys(keyList=["escape"]):
                        core.quit()

win.close()

if ok_data[5] == 'Execution':
    end_dlg = gui.Dlg(title=expName)
    end_dlg.addText('Stop LSL before clicking OK', color='red')
    end_dlg.show()  # show dialog and wait for OK or Cancel

    load_xdf(experiment_path + '/untitled.xdf')
core.quit()