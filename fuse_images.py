# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:27:18 2018

@author: sofha
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data_path='C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
background = Image.open(os.path.join(data_path, '00001_930831_fb_a.pgm'), mode='r')
foreground = Image.open(os.path.join(data_path,'00003_940307_fb_a.pgm'))
# set alpha to .7
i1=Image.blend(background, foreground, .2)#.save("out1.bmp")#show()#
i1.show()