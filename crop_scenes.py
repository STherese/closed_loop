# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 07:27:05 2018

@author: sofha
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
i=20
data_path_faces='C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/indoor/0/pgms'
face_images=os.listdir(data_path_faces)
image_face = Image.open(os.path.join(data_path_faces,face_images[i]))
image_face.show()
scenes_images=os.listdir(data_path_scenes)
data_path_scenes_outdoor='C:/no_backup/no_backup/closed_loop/scenes/outdoor/0/pgms'
scenes_outdoor_images=os.listdir(data_path_scenes_outdoor)
tstart=time.time()
image_scene = Image.open(os.path.join(data_path_scenes,scenes_images[i]))
image_scene.show()
time_pgm=time.time()-tstart
size = 130,130
def resize_image(size,im):
#ratio=130/130
    h,w=im.size
    im_ratio=w/h
    if w>h:
        size_w=int(im_ratio*size[0])
        size_new=size[0],size_w
        im_resized = im.resize(size_new, Image.ANTIALIAS)


    if w<=h:
        size_h=int(size[1]/im_ratio)
        size_new=size_h,size[1]
        im_resized = im.resize(size_new, Image.ANTIALIAS)
    
    return im_resized
    
im_scene_resized = resize_image(size,image_scene)
im_scene_cropped = im_scene_resized.crop((0, 0, 130, 130))
width=260
height=260
im_scenes = im_scene_cropped.resize((width, height), Image.NEAREST)   
im_faces_cropped = image_face.crop((0, 0, 130, 130))
im_faces = im_faces_cropped.resize((width, height), Image.NEAREST)  
bin_edges = 0, 55, 200, 255
quantiles = 0, 0.2, 0.95, 1.0
from image_norm_bins import hist_norm
im_faces = hist_norm(np.array(im_faces,'float64'), bin_edges, quantiles, inplace=False)
im_faces=Image.fromarray(im_faces.astype('uint8'),mode='L')
im_scenes = hist_norm(np.array(im_scenes,'float64'), bin_edges, quantiles, inplace=False)
im_scenes=Image.fromarray(im_scenes.astype('uint8'),mode='L')
scenes_outdoor_dest="C:/no_backup/no_backup/closed_loop/stimuli/scenes/outdoor"
faces_dest="C:/no_backup/no_backup/closed_loop/stimuli/faces/"

#im_scenes.save("C:/Users/sofha/Documents/closed_loop/closed_loop/stimuli/scenes/"+"scene"+str(i)+".jpg", "JPEG")
%im_scenes.save(scenes_outdoor_dest+"scene"+str(i)+".jpg", "JPEG")

im_faces.save(faces_dest+"face"+str(i)+".jpg", "JPEG")

#im_faces.save("C:/Users/sofha/Documents/closed_loop/closed_loop/stimuli/faces/"+"face"+str(i)+".jpg", "JPEG")

#%%
#im_cropped.show()
i1=Image.blend(im_faces, im_scenes, 0.5)#.save("out1.bmp")#show()#
#plt.imshow(im_faces, im_scenes, alpha=0.5)
i1.show()
#%%
import operator
import functools 

def equalize(im):
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        print(b)
        # step size
        step = functools.reduce(operator.add, h[b:b+256]) / 175#255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut*1)

im_eq=equalize(im_faces)
im_eq.show()
#%%
bin_edges = 0, 55, 200, 255
quantiles = 0, 0.2, 0.95, 1.0
from image_norm_bins import hist_norm
im_faces_mean=np.zeros(500)
im_faces_median=np.zeros(500)
im_faces_std=np.zeros(500)
for i in range(500):
    image_face = Image.open(os.path.join(data_path_faces,face_images[i]))
    im_faces = image_face.crop((0, 0, 130, 130)) 
    im_faces = hist_norm(np.array(im_faces,'float64'), bin_edges, quantiles, inplace=False)
    image_fft = np.fft.fft2(im_faces)
    im_real=np.real(image_fft)
    im_real[0,0]=np.mean(im_real)
#plt.imshow(np.log(np.abs(im_real)))
    image_fft=np.abs(image_fft)
    image_fft_rav=image_fft.ravel()
    image_fft_rav=image_fft_rav[np.nonzero(image_fft_rav)]
    im_faces_mean[i]=np.mean(image_fft_rav)
    im_faces_median[i]=np.median(image_fft_rav)

    im_faces_std[i]=np.std(image_fft_rav)
    #plt.figure(12),plt.imshow(im_faces,cmap=plt.cm.gray)
    #plt.pause(1)
    #plt.figure(6)
    #plt.hist(np.log(image_fft_rav), bins=100)
    
#%%
im_scenes_mean=np.zeros(500)
im_scenes_median=np.zeros(500)
data_path_scenes_outdoor='C:/no_backup/no_backup/closed_loop/scenes/outdoor/0/pgms'
scenes_outdoor_images=os.listdir(data_path_scenes_outdoor)

im_scenes_std=np.zeros(500)
for i in range(500):
    image_scene = Image.open(os.path.join(data_path_scenes_outdoor,scenes_outdoor_images[i]))
    im_scene_resized = resize_image(size,image_scene)

    im_cropped = im_scene_resized.crop((0, 0, 130, 130))   
    im_cropped = hist_norm(np.array(im_cropped,'float64'), bin_edges, quantiles, inplace=False)

    image_fft = np.fft.fft2(im_cropped)
#plt.imshow(np.log(np.abs(im_real)))
    image_fft=np.abs(image_fft)
    image_fft_rav=image_fft.ravel()
    image_fft_rav=image_fft_rav[np.nonzero(image_fft_rav)]
    #plt.figure(10)
    #plt.hist(np.log(image_fft_rav), bins=100)
    im_scenes_mean[i]=np.mean(image_fft_rav)
    im_scenes_median[i]=np.median(image_fft_rav)
    im_scenes_std[i]=np.std(image_fft_rav)
    #plt.figure(12),plt.imshow(im_cropped,cmap=plt.cm.gray)
    #plt.pause(0.5)
    
#%% alpha = 0.5
plt.figure(11)
plt.hist(im_faces_median, bins=50,alpha = 0.5)
plt.hist(im_scenes_median, bins=50,alpha = 0.5)

plt.figure(21)
plt.hist(im_faces_mean, bins=50,alpha = 0.5)
plt.hist(im_scenes_mean, bins=50,alpha = 0.5)
plt.figure(31)
plt.hist(im_faces_std, bins=50,alpha = 0.5)
plt.hist(im_scenes_std, bins=50,alpha = 0.5)

#%%
bin_edges = 0, 55, 200, 255
quantiles = 0, 0.2, 0.95, 1.0
img = np.array(image_scene,'float64')#lena()
normed = hist_norm(img, bin_edges, quantiles)
#Plotting:

from matplotlib import pyplot as plt

def ecdf(x):
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

x1, y1 = ecdf(img.ravel())
x2, y2 = ecdf(normed.ravel())

fig = plt.figure()
gs = plt.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[1, :])
for aa in (ax1, ax2):
    aa.set_axis_off()

ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Original')
ax2.imshow(normed, cmap=plt.cm.gray)
ax2.set_title('Normalised')

ax3.plot(x1, y1 * 100, lw=2, label='Original')
ax3.plot(x2, y2 * 100, lw=2, label='Normalised')
for xx in bin_edges:
    ax3.axvline(xx, ls='--', c='k')
for yy in quantiles:
    ax3.axhline(yy * 100., ls='--', c='k')
ax3.set_xlim(bin_edges[0], bin_edges[-1])
ax3.set_xlabel('Pixel value')
ax3.set_ylabel('Cumulative %')
ax3.legend(loc=2)
#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 07:27:05 2018

@author: sofha
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

def resize_crop_norm(im):
    size = 130,130
    im_resized = resize_image(size,im)
    im_cropped = im_resized.crop((0, 0, 130, 130))
    width=225
    height=225
    im_new = im_cropped.resize((width, height), Image.NEAREST)   
    bin_edges = 0, 55, 200, 255
    quantiles = 0, 0.2, 0.95, 1.0
    from image_norm_bins import hist_norm
    im_norm = hist_norm(np.array(im_new,'float64'), bin_edges, quantiles, inplace=False)
    im_norm=Image.fromarray(im_norm.astype('uint8'),mode='L')
    return im_norm


def resize_crop_norm_edges(im):
    size = 130,130
    im_resized = resize_image(size,im)
    im_cropped = im_resized.crop((0, 0, 130, 130))
    width=225
    height=225
    bin_edges = 0,46,210,255#0,59, 199,255
    quantiles = 0,0.1, 0.90,1.0
    from image_norm_bins import hist_norm
    im_norm = hist_norm(np.array(im_cropped,'float64'), bin_edges, quantiles, inplace=False)
    im_norm=Image.fromarray(im_norm.astype('uint8'),mode='L')
    #bins=np.arange(0, 255, 10)
    freq, bins = np.histogram(im_cropped.crop((15,15,115,115)),256)
    im_new = im_norm.resize((width, height), Image.LANCZOS)   
    return im_new, freq

data_path_faces='C:/no_backup/no_backup/closed_loop/dvd2/gray_feret_cd1/data/images/cropped_auto'#'C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/indoor/0/'
scene_images=os.listdir(data_path_scenes)
face_images=os.listdir(data_path_faces)
data_path_scenes_outdoor='C:/no_backup/no_backup/closed_loop/scenes/outdoor/0/pgms'
faces_dest='C:/no_backup/no_backup/closed_loop/stimuli_edges/faces/female/'
scenes_dest='C:/no_backup/no_backup/closed_loop/stimuli_edges/scenes/indoor/'

#Freq=np.zeros([1330,256])
#FreqFaces=np.zeros([3672,256])
#FreqScenesIndoor=np.zeros([3151,256])
import scipy.io as sio
n=sio.loadmat(os.path.join(faces_dest,'numbers.mat'))
N=n['numbers_cd1']
#N=N-2342
for ii in range((N.size)):#range(len(face_images))
    i=N[0,ii]
    im = Image.open(os.path.join(data_path_faces,face_images[i])).convert('L')
    im_norm,freq=resize_crop_norm_edges(im)
    #FreqFaces[i+2342,:]=freq
    #FreqScenesIndoor[i,:]=freq
    c=113
    for iii in range(-12,12):
        for ii in range(-2,2): 
            im_norm.putpixel((c+ii,c+iii),255)
    for iii in range(-12,12):
        for ii in range(-2,2): 
            im_norm.putpixel((c+iii,c+ii),255)
    im_norm.save(faces_dest+"face"+str(i)+".jpg", "JPEG")

#%%
scenes_dest='C:/Users/sofha/Documents/closed_loop/closed_loop/stimuli/scenes/indoor/'#'C:/no_backup/no_backup/closed_loop/stimuli_edges/scenes/outdoor/'
#data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/outdoor/0/'
data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/indoor/lobby/'
#n=sio.loadmat(os.path.join(scenes_dest,'numbers.mat'))
#N=n['numbers']
scene_images=os.listdir(data_path_scenes)
for i in range(len(scene_images)):#ii in range((N.size)):#
    #i=N[0,ii]
    im = Image.open(os.path.join(data_path_scenes,scene_images[i])).convert('L')
    im_norm,freq=resize_crop_norm_edges(im)
    #FreqFaces[i+2342,:]=freq
    #FreqScenesIndoor[i,:]=freq
    c=113
    for iii in range(-12,12):
        for ii in range(-2,2): 
            im_norm.putpixel((c+ii,c+iii),255)
    for iii in range(-12,12):
        for ii in range(-2,2): 
            im_norm.putpixel((c+iii,c+ii),255)
    im_norm.save(scenes_dest+"scene"+str(i+3150)+".jpg", "JPEG")
#%%
freq_mean=np.mean(FreqFaces,axis=0)
freq_cum=np.cumsum(freq_mean)/sum(freq_mean)
plt.plot(freq_cum)
freq1=np.argsort(abs(freq_cum-0.01))
freq10=np.argsort(abs(freq_cum-0.10))
freq90=np.argsort(abs(freq_cum-0.90))
freq99=np.argsort(abs(freq_cum-0.99))
#%%

freq_mean=np.mean(FreqScenesOutdoor,axis=0)
freq_cum_out=np.cumsum(freq_mean)/sum(freq_mean)
plt.plot(freq_cum)
freq1_out=np.argsort(abs(freq_cum_out-0.01))
freq10_out=np.argsort(abs(freq_cum_out-0.10))
freq90_out=np.argsort(abs(freq_cum_out-0.90))
freq99_out=np.argsort(abs(freq_cum_out-0.99))

#%%
freq_mean=np.mean(FreqScenesIndoor,axis=0)
freq_cum_in=np.cumsum(freq_mean)/sum(freq_mean)
plt.plot(freq_cum)
freq1_in=np.argsort(abs(freq_cum_in-0.01))
freq10_in=np.argsort(abs(freq_cum_in-0.10))
freq90_in=np.argsort(abs(freq_cum_in-0.90))
freq99_in=np.argsort(abs(freq_cum_in-0.99))
#%%
FREQ1=(freq1[0]+freq1_in[0]+freq1_out[0])/3
FREQ10=(freq10[0]+freq10_in[0]+freq10_out[0])/3
FREQ90=(freq90[0]+freq90_in[0]+freq90_out[0])/3
FREQ99=(freq99[0]+freq99_in[0]+freq99_out[0])/3
#%%
faces_dest='C:/no_backup/no_backup/closed_loop/stimuli/faces/'

face_images=os.listdir(faces_dest)
n_faces=len(face_images)
im_faces_mean=np.zeros(n_faces)
im_faces_median=np.zeros(n_faces)
im_faces_std=np.zeros(n_faces)
im_faces_above=np.zeros(n_faces)
im_faces_below=np.zeros(n_faces)
im_faces_large=np.zeros(n_faces)
plt.figure(1)

for i in range(n_faces):
    im_faces = Image.open(os.path.join(faces_dest,face_images[i]))
    #im_faces = image_face.crop((0, 0, 130, 130)) 
    #im_faces = hist_norm(np.array(im_faces,'float64'), bin_edges, quantiles, inplace=False)
    image_fft = np.fft.fft2(im_faces)
    im_real=np.real(image_fft)
    im_real[0,0]=np.mean(im_real)
#plt.imshow(np.log(np.abs(im_real)))
    image_fft=np.abs(image_fft)
    image_fft_rav=image_fft.ravel()
    image_fft_rav=image_fft_rav[np.nonzero(image_fft_rav)]
    #im_faces_above[i]=sum( image_fft_rav> 2000)
    im_faces_below[i]=sum( image_fft_rav< 2000)
    im_faces_large[i]=sum( image_fft_rav> 10000)
    im_faces_mean[i]=np.mean(image_fft_rav)
    im_faces_median[i]=np.median(image_fft_rav)
    im_faces_std[i]=np.std(image_fft_rav)
    #plt.hist(image_fft_rav, bins=500,alpha = 0.5)
#%%
scenes_dest='C:/no_backup/no_backup/closed_loop/stimuli/scenes/outdoor/'
scene_images=os.listdir(scenes_dest)
n_scenes=len(scene_images)
im_scenes_mean=np.zeros(n_scenes)
im_scenes_median=np.zeros(n_scenes)
im_scenes_std=np.zeros(n_scenes)
im_scenes_above=np.zeros(n_scenes)
im_scenes_below=np.zeros(n_scenes)
im_scenes_large=np.zeros(n_scenes)
for i in range(n_scenes):
    im_scenes = Image.open(os.path.join(scenes_dest,scene_images[i]))
    #im_faces = image_face.crop((0, 0, 130, 130)) 
    #im_faces = hist_norm(np.array(im_faces,'float64'), bin_edges, quantiles, inplace=False)
    image_fft = np.fft.fft2(im_scenes)
    im_real=np.real(image_fft)
    im_real[0,0]=np.mean(im_real)
#plt.imshow(np.log(np.abs(im_real)))
    image_fft=np.abs(image_fft)
    image_fft_rav=image_fft.ravel()
    image_fft_rav=image_fft_rav[np.nonzero(image_fft_rav)]
    #im_scenes_above[i]=sum( image_fft_rav> 2000)
    im_scenes_below[i]=sum( image_fft_rav< 2000)
    im_scenes_large[i]=sum( image_fft_rav> 10000)
    im_scenes_mean[i]=np.mean(image_fft_rav)
    im_scenes_median[i]=np.median(image_fft_rav)
    im_scenes_std[i]=np.std(image_fft_rav)

#%%
scenes_dest='C:/no_backup/no_backup/closed_loop/stimuli/scenes/indoor/'
scene_images=os.listdir(scenes_dest)
n_scenes=len(scene_images)
im_scenes_indoor_mean=np.zeros(n_scenes)
im_scenes_indoor_median=np.zeros(n_scenes)
im_scenes_indoor_std=np.zeros(n_scenes)
im_scenes_indoor_above=np.zeros(n_scenes)
im_scenes_indoor_below=np.zeros(n_scenes)
im_scenes_indoor_large=np.zeros(n_scenes)
for i in range(n_scenes):
    im_scenes = Image.open(os.path.join(scenes_dest,scene_images[i]))
    #im_faces = image_face.crop((0, 0, 130, 130)) 
    #im_faces = hist_norm(np.array(im_faces,'float64'), bin_edges, quantiles, inplace=False)
    image_fft = np.fft.fft2(im_scenes)
    im_real=np.real(image_fft)
    im_real[0,0]=np.mean(im_real)
#plt.imshow(np.log(np.abs(im_real)))
    image_fft=np.abs(image_fft)
    image_fft_rav=image_fft.ravel()
    image_fft_rav=image_fft_rav[np.nonzero(image_fft_rav)]
    #im_scenes_indoor_above[i]=sum( image_fft_rav> 2000)
    im_scenes_indoor_below[i]=sum( image_fft_rav< 2000)
    im_scenes_indoor_large[i]=sum( image_fft_rav> 10000)
    im_scenes_indoor_mean[i]=np.mean(image_fft_rav)
    im_scenes_indoor_median[i]=np.median(image_fft_rav)
    im_scenes_indoor_std[i]=np.std(image_fft_rav)
#%%
plt.figure(12)
plt.hist(im_faces_median, bins=50,alpha = 0.5)
plt.hist(im_scenes_median, bins=50,alpha = 0.5)
plt.hist(im_scenes_indoor_median, bins=50,alpha = 0.5)
plt.title('Median frequency')
plt.legend(['Faces','Outdoor','Indoor'])
plt.figure(22)
plt.hist(im_faces_mean, bins=50,alpha = 0.5)
plt.hist(im_scenes_mean, bins=50,alpha = 0.5)
plt.hist(im_scenes_indoor_mean, bins=50,alpha = 0.5)
plt.title('Mean frequency')
plt.legend(['Faces','Outdoor','Indoor'])
plt.figure(32)
plt.hist(im_faces_std, bins=50,alpha = 0.5)
plt.hist(im_scenes_std, bins=50,alpha = 0.5)
plt.hist(im_scenes_indoor_std, bins=50,alpha = 0.5)
plt.title('STD of frequency')
plt.legend(['Faces','Outdoor','Indoor'])
#%%
plt.figure(3)
plt.hist(im_faces_below, bins=50,alpha = 0.5)
plt.hist(im_scenes_below, bins=50,alpha = 0.5)
plt.hist(im_scenes_indoor_below, bins=50,alpha = 0.5)
plt.figure(4)
plt.hist(im_faces_large, bins=50,alpha = 0.5)
plt.hist(im_scenes_large, bins=50,alpha = 0.5)
plt.hist(im_scenes_indoor_large, bins=50,alpha = 0.5)
#%%
std_faces_below=np.std(im_faces_below)
std_faces_large=np.std(im_faces_large)
mean_faces_below=np.mean(im_faces_below)
mean_faces_large=np.mean(im_faces_large)
c=0
faces_keep=np.empty((n_faces))
faces_keep[:] = np.nan
for i in range(n_faces):
    if abs(im_faces_large[i]-mean_faces_large)<3*std_faces_large and abs(im_faces_below[i]-mean_faces_below)<3*std_faces_below:
        faces_keep[i]=1
    else:
        faces_keep[i]=0
        
#%%
#sorted(im_scenes_median-750)
sort_index = np.argsort(abs(im_scenes_std))
im_scenes = Image.open(os.path.join(scenes_dest,scene_images[sort_index[-1]]))
im_scenes.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

data_path_faces='C:/no_backup/no_backup/closed_loop/dvd2/gray_feret_cd1/data/images/cropped_auto'#'C:/no_backup/no_backup/closed_loop/faces/normalised_faces/cropped'
data_path_scenes='C:/no_backup/no_backup/closed_loop/scenes/indoor/0/'
scene_images=os.listdir(data_path_scenes)
face_images=os.listdir(data_path_faces)
data_path_scenes_outdoor='C:/no_backup/no_backup/closed_loop/scenes/outdoor/0/pgms'
faces_dest='C:/no_backup/no_backup/closed_loop/stimuli_edges/faces/'
scenes_dest='C:/no_backup/no_backup/closed_loop/stimuli_edges/scenes/indoor/'

#Freq=np.zeros([1330,256])
#FreqFaces=np.zeros([3672,256])
#FreqScenesIndoor=np.zeros([3151,256])
def resize_image(size,im):
#ratio=130/130
    h,w=im.size
    im_ratio=w/h
    if w>h:
        size_w=int(im_ratio*size[0])
        size_new=size[0],size_w
        im_resized = im.resize(size_new, Image.ANTIALIAS)


    if w<=h:
        size_h=int(size[1]/im_ratio)
        size_new=size_h,size[1]
        im_resized = im.resize(size_new, Image.ANTIALIAS)
    
    return im_resized

def resize_crop_norm_opt(im):
    size = 130,130
    im_resized = resize_image(size,im)
    im_cropped = im_resized.crop((0, 0, 130, 130))
    width=225
    height=225
    bin_edges = 0,46,210,255#0,59, 199,255
    quantiles = 0,0.1, 0.90,1.0
    from image_norm_bins import hist_norm
    im_norm = hist_norm(np.array(im_cropped,'float64'), bin_edges, quantiles, inplace=False)
    im_norm=Image.fromarray(im_norm.astype('uint8'),mode='L')
    #bins=np.arange(0, 255, 10)
    freq, bins = np.histogram(im_cropped.crop((15,15,115,115)),256)
    return im_norm, freq

width=225
height=225
for i in range(2,len(face_images)):
    im = Image.open(os.path.join(data_path_scenes,scene_images[i])).convert('L')
    im_norm,freq=resize_crop_norm_opt(im)
    im_new = im_norm#.resize((width, height), Image.NEAREST)   
    im_new.show()
    im_nearest = im_norm.resize((width, height), Image.NEAREST) 
    im_nearest.show()
    im_bilinear = im_norm.resize((width, height), Image.BILINEAR)
    im_bilinear.show()
    im_bicubic = im_norm.resize((width, height), Image.BICUBIC) 
    im_bicubic.show()
    im_lan = im_norm.resize((width, height), Image.LANCZOS) 
    im_lan.show()