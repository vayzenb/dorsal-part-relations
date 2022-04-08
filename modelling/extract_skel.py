# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:10:51 2021

@author: vayze
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#import cv2 as cv
from skimage import io
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.segmentation import chan_vese
from skimage.transform import resize
#import matplotlib.image as mpimg
import math
from PIL import Image
import scipy.ndimage.morphology as morphOps
from skimage.filters import  gaussian
from skimage.util import crop
from skimage.util import pad
import os
import glob as glob
import pdb


stim_folder = "/home/vayzenbe/GitHub_Repos/docnet/stim/binary"
out_folder = "/home/vayzenbe/GitHub_Repos/docnet/stim/"

stim_folder = "C:/Users/vayze/Desktop/GitHub_Repos/docnet/stim/binary"
out_folder = "C:/Users/vayze/Desktop/GitHub_Repos/docnet/stim"
skel = ['23', '26', '31', '31_0', '31_50']
SF = ['Skel','Bulge']

pad_num = 50

            # In[73]:
def sample_sphere_2D(number_of_samples):
    sphere_points = np.zeros((number_of_samples,2))
    alpha = (2*math.pi)/(number_of_samples)
    for i in range(number_of_samples):
        sphere_points[i][0] = math.cos(alpha*(i-1))
        sphere_points[i][1] = math.sin(alpha*(i-1))
    return sphere_points


def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def compute_aof(distImage ,IDX,sphere_points,epsilon):

    m = distImage.shape[0]
    n = distImage.shape[1]
    normals = np.zeros(sphere_points.shape)
    fluxImage = np.zeros((m,n))
    for t in range(0,number_of_samples):
        normals[t] = sphere_points[t]
    sphere_points = sphere_points * epsilon
    
    XInds = IDX[0]
    YInds = IDX[1]
    
    for i in range(0,m):
        #print(i)
        for j in range(0,n):       
            flux_value = 0
            if (distImage[i][j] > -1.5):
                if( i > epsilon and j > epsilon and i < m - epsilon and j < n - epsilon ):
#                   sum over dot product of normal and the gradient vector field (q-dot)
                    for ind in range (0,number_of_samples):
                                                
#                       a point on the sphere
                        px = i+sphere_points[ind][0]+0.5;
                        py = j+sphere_points[ind][1]+0.5;
                        
                        
                        
                        
#                       the indices of the grid cell that sphere points fall into 
                        cI = math.floor(i+sphere_points[ind][0]+0.5)
                        cJ = math.floor(j+sphere_points[ind][1]+0.5)
                                               

#                       closest point on the boundary to that sphere point

                        bx = XInds[cI][cJ]
                        by = YInds[cI][cJ]
#                       the vector connect them
                        qq = [bx-px,by-py]
                    
                        d = np.linalg.norm(qq)
                        if(d!=0):
                            qq = qq / d
                        else:
                            qq = [0,0]                        
                        flux_value = flux_value + np.dot(qq,normals[ind])
            fluxImage[i][j] = flux_value  
    return fluxImage



#imfiles = glob.glob(f'{stim_folder}/*.png')
imfiles = ['C:/Users/vayze/Documents/Research/Presentations/Final Presentations/dog_binary.png']
#os.makedirs(f'{out_folder}/binary/', exist_ok = True)
os.makedirs(f'{out_folder}/blur/', exist_ok = True)
os.makedirs(f'{out_folder}/coords/', exist_ok = True)


#jit(nopython=True)
for imf in imfiles:

    #inframe = f'{stim_folder}/Figure_{sk}_{sf}/Figure_{sk}_{sf}_{ff}.jpg'
    #figure= f'Figure_{sk}_{sf}/Figure_{sk}_{sf}_{ff}'
    curr_file = imf.replace(stim_folder + '/', '')
    curr_file = curr_file[:-4]
    
    
    print(curr_file)
    im = io.imread(imf)
    

    im = rgb2gray(im)
    im = np.pad(im, pad_num) 
    im = resize(im, [225 +pad_num,225 +pad_num], anti_aliasing=True)

    #thresh = threshold_otsu(im)
    #binary = im > thresh
    
    
    img = img_as_float(im)
    filtered_img = gaussian(img, sigma=2)
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(img, mu=0.25, lambda1=.5, lambda2=1, tol=1e-3, max_iter=200,
                dt=0.5, init_level_set='checkerboard', extended_output=True)
    
    
    silh = cv[0].astype(int)
    '''
    if np.mean(silh[0:25, 0:25]) > 0:
        cv = chan_vese(filtered_img, mu=0.1, lambda1=.5, lambda2=1, tol=1e-3, max_iter=500,
        dt=.5, init_level_set='checkerboard', extended_output=True)
        silh = cv[0].astype(int)
    '''
    
    I = silh
    #print(I.shape)
    
    number_of_samples = 60
    epsilon = 1 
    flux_threshold = 18

    
    # In[73]:
    
    distImage,IDX = morphOps.distance_transform_edt(I,return_indices=True);
    sphere_points = sample_sphere_2D(number_of_samples)
    fluxImage = compute_aof(distImage,IDX,sphere_points,epsilon)
    
    
    
    # In[74]:
    
    
    #print(fluxImage.shape)
    
    
    # In[75]:
    
    
    #plt.imshow(fluxImage)
    skeletonImage = fluxImage
    
    
    skel_coords = np.column_stack(np.where(skeletonImage > flux_threshold))
    skel_vals = skeletonImage[skel_coords[:,0], skel_coords[:,1]]
    skel = np.zeros((len(skel_vals),3))

    skel[:,:2] = skel_coords
    skel[:,2] = skel_vals
    #skel_coords = skeletonImage[skeletonImage > flux_threshold]
    
    skeletonImage[skeletonImage < flux_threshold] = 0
    skelim = np.interp(skeletonImage, (skeletonImage.min(), skeletonImage.max()), (0, 255))
    #skelim = skelim[pad_num:(skelim.shape[0]-pad_num), pad_num:(skelim.shape[1]-pad_num)]
    
    #plt.imshow(skelim, cmap="gray")
    #save coords
    np.savetxt(f'{out_folder}/coords/{curr_file}.csv', skel,delimiter=',')
    #save binary
    skelim = crop(skelim, 5)
    #io.imsave(f'{out_folder}/binary/{curr_file}.jpg',skelim)

    #save blurred binary
    skelim = gaussian(skelim, sigma=3)
    
    io.imsave(f'{out_folder}/blur/{curr_file}.jpg',skelim)
    #skeletonImage[skeletonImage > flux_threshold] = 1
