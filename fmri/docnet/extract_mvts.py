"""
Extract multivariate timeseries from ROIs
for use in multivariate granger cauality

PCA is applied to the extracted TS and saved as MAT file

"""
# %%

import warnings
warnings.filterwarnings("ignore")
import resource
import sys
import time
import os
import gc
import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA

from nilearn import image, datasets
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball
import scipy.io as spio

print('libraries loaded...')



#load subj number and seed
#subj
#ss = int(sys.argv[1])
#seed region
#dorsal = str(sys.argv[2])
study ='docnet'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
exp = 'catmvpa'
'''
print(ss, dorsal)


out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/vayzenbe/GitHub_Repos/docnet/results'

sub_dir = f'{study_dir}/sub-{study}{ss}/ses-02/'
cov_dir = f'{sub_dir}/covs'
roi_dir = f'{sub_dir}/derivatives/rois'
exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
'''
runs = list(range(1,9))
#runs = list(range(1,3))

#whole_brain_mask = datasets.load_mni152_brain_mask()
whole_brain_mask = image.load_img('/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz')
affine = whole_brain_mask.affine
dimsize = whole_brain_mask.header.get_zooms()  #get dimenisions

#
vols = 331
first_fix = 8

pc_thresh = .9

"""
Setup searchlight
"""
print('Searchlight setup ...')
#set search light params
#data = bold_vol #data as 4D volume (in numpy)
mask = image.get_data(whole_brain_mask) #the mask to search within
#mask = np.zeros(whole_brain_mask.shape) + 1
#mask = mask * image.get_data(whole_brain_mask)



# %%
def extract_pc(data, n_components=None):

    """
    Extract principal components
    if n_components isn't set, it will extract all it can
    """
    
    pca = PCA(n_components = n_components)
    pca.fit(data)
    
    return pca

# %%
def calc_pc_n(pca, thresh):
    '''
    Calculate how many PCs are needed to explain X% of data
    
    pca - result of pca analysis
    thresh- threshold for how many components to keep
    '''
    
    explained_variance = pca.explained_variance_ratio_
    
    var = 0
    for n_comp, ev in enumerate(explained_variance):
        var += ev #add each PC's variance to current variance
        #print(n_comp, ev, var)

        if var >=thresh: #once variance > than thresh, stop
            break
    
    '''
    plt.bar(range(len(explained_variance[0:n_comp+1])), explained_variance[0:n_comp+1], alpha=0.5, align='center')
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()
    '''
    return n_comp+1



# %%
def load_data(sub):
    sub_dir = f'{study_dir}/sub-{study}{sub}/ses-02/'
    exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
    print('Loading data...')

    all_runs = []
    for run in runs:
        print(run)

        curr_run = image.load_img(f'{exp_dir}/run-0{run}/1stLevel.feat/filtered_func_data_reg.nii.gz') #load data
        curr_run = image.get_data(image.clean_img(curr_run,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy
        curr_run = curr_run[:,:,:,first_fix:] #remove first few fixation volumes

        all_runs.append(curr_run) #append to list


        del curr_run
        print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)

    print('data loaded..')

    print('concatenating data..')
    bold_vol = np.concatenate(np.array(all_runs),axis = 3) #compile into 4D
    del all_runs
    print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
    print('data concatenated...')
    gc.collect()

    return bold_vol



    # %%
def extract_ts(bold_vol, roi):
    #extract all data from seed

    #load seed
    print('Extracting TS...')
    seed_roi = image.get_data(image.load_img(f'{roi_dir}/spheres/{roi}_sphere.nii.gz'))
    reshaped_roi = np.reshape(seed_roi, (91,109,91,1))
    masked_img = reshaped_roi*bold_vol

    #extract voxel resposnes from within mask
    seed_ts = masked_img.reshape(-1, bold_vol.shape[3]) #reshape into rows (voxels) x columns (time)
    seed_ts =seed_ts[~np.all(seed_ts == 0, axis=1)] #remove voxels that are 0 (masked out)
    seed_ts = np.transpose(seed_ts)

    print('Seed data extracted...')

    return seed_ts

# %%
subj_list = [2001,2002,2003,2004, 2005, 2007, 2008, 2012, 2013, 2014, 2015, 2016]
subj_list = [2017, 2018]
rois =  ['PPC_spaceloc',   'APC_spaceloc',  'APC_distloc', 'LO_toolloc']
for ss in subj_list:
    sub_dir = f'{study_dir}/sub-{study}{ss}/ses-02/'
    exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
    cov_dir = f'{sub_dir}/covs'
    roi_dir = f'{sub_dir}/derivatives/rois'
    results_dir = f'{sub_dir}/derivatives/results/beta_ts'
    os.makedirs(f'{sub_dir}/derivatives/results/beta_ts', exist_ok=True)
    bold_vol = load_data(ss)
    for lr in ['l','r']:
        for rr in rois:
            if os.path.exists(f'{roi_dir}/{lr}{rr}.nii.gz'):
                seed_ts = extract_ts(bold_vol, f'{lr}{rr}')
                n_comp = calc_pc_n(extract_pc(seed_ts),.9)
                seed_pca = extract_pc(seed_ts, n_comp) #conduct PCA one more time with that number of PCs
                #print(seed_pca.shape)

                seed_pcs = seed_pca.transform(seed_ts) #transform train data in PCs
                seed_pcs = seed_pcs.reshape((1,seed_pcs.shape[0],seed_pcs.shape[1]))
                spio.savemat(f'{results_dir}/{lr}{rr}_pc_ts.mat', {f'{lr}{rr}_pc_ts': seed_pcs})
                print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
                print(ss, f'{lr}{rr}')
    del bold_vol
    gc.collect()








