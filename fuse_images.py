# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 10:27:18 2018

@author: sofha
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

data_path='C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
background = Image.open(os.path.join(data_path, '00001_930831_fb_a.pgm'), mode='r')
tstart=time.time()
foreground = Image.open(os.path.join(data_path,'00003_940307_fb_a.pgm'))
time_pgm=time.time()-tstart
# set alpha to .7
i1=Image.blend(background, foreground, .2)#.save("out1.bmp")#show()#
i1.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

data_path_faces='C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/indoor/0/pgms'
face_images=os.listdir(data_path_faces)
background = Image.open(os.path.join(data_path_faces,face_images[1]))
scenes_images=os.listdir(data_path_scenes)
s=0;
while s <10:#len(scenes_images)
    tstart=time.time()
    foreground = Image.open(os.path.join(data_path_scenes,scenes_images[1]))
    time_pgm=time.time()-tstart
    size = 130,160
    ratio=130/160
    w,h=foreground.size
    im_ratio=w/h
    if im_ratio>ratio:
        size_w=int(ratio*h)
        size_new=size_w,160
        im_resized = foreground.resize(size_new, Image.ANTIALIAS)


    if im_ratio<ratio:
        size_h=int(1/ratio*w)
        size_new=130,size_h
        im_resized = foreground.resize(size_new, Image.ANTIALIAS)

    im_cropped = im_resized.crop((0, 0, 130, 160))
    data_path_scenes_save='C:/no_backup/no_backup/closed_loop/scenes/indoor/0/pgms/transformed/'
    ext='.pgm'
    im_cropped.save(data_path_scenes_save+'scene'+ str(s) + ext)
    s=s+1
#%%
width=130#260
height=160#320
im_scenes = im_cropped.resize((width, height), Image.NEAREST)   
im_faces = background.resize((width, height), Image.NEAREST)  
#im_cropped.show()
i1=Image.blend(im_faces, im_scenes, .5)#.save("out1.bmp")#show()#
i1.show()