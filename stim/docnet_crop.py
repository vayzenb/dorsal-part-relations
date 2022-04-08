# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:58:05 2021

@author: vayze
"""
import os

from PIL import Image
import numpy as np
from glob import glob
import pdb

im_dir = 'obj_images'
im_files = [y for x in os.walk(f'{im_dir}/original') for y in glob(os.path.join(x[0], '*.jpg'))]

classes = os.listdir(im_dir)


im_buffer = 10
n = 1
for imn, imf in enumerate(im_files):
    og_im = Image.open(imf).convert("RGB")
    
    im = np.array(og_im)
    
    #Find where the red is by indexing the non-gray colors in a channel
    locs = np.argwhere(im[:,:,1] < 85)
    
    #find left, top, right, and bottom
    top = np.min(locs[:,0])
    left = np.min(locs[:,1])
    bottom = np.max(locs[:,0])
    right = np.max(locs[:,1])
    
    
    #Crop IM
    crop_im = og_im.crop((left-im_buffer, top-im_buffer,  right+im_buffer, bottom+im_buffer))
    
    #determine size of new images
    im_size = np.max(crop_im.size)
    
    #Make a new image as a square
    new_im = Image.new(crop_im.mode, [im_size, im_size],(89, 89, 89))
    
    #Determine where to paste it
    t_loc = int((im_size - crop_im.size[0])/2)
    l_loc = int((im_size- crop_im.size[1])/2)
    
    #paste into new_im
    new_im.paste(crop_im, (t_loc, l_loc)) 
        
    #doing this twice
    filename = imf[imf.index('\\')+1:]
    filename = filename[filename.index('\\')+1:]
    
    #check if same exemplar and category
    if imn == 0: #for the very first filename, just set an initial prev_file
        prev_file = filename.split('_') 
        curr_file = prev_file
    else: #for all others, check if the category and exemplar are the same
        curr_file = filename.split('_') 
        
        #if the cat is the same, but exemplar is different, set increase n
        if prev_file[0] == curr_file[0] and prev_file[1] != curr_file[1]:
            n += 1
        elif prev_file[0] != curr_file[0]: #if cat has changed, reset n and set new prev_file
            n = 1
        
        prev_file = filename.split('_') 
        
            
    
    new_file = f'{curr_file[0]}_{n}_{curr_file[2]}'
    #pdb.set_trace()

    new_im.save(f'{im_dir}/cropped/{new_file}')

