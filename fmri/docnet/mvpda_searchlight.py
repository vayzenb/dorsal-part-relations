# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression

from nilearn import image, datasets
import nibabel as nib
from brainiak.searchlight.searchlight import Searchlight, Ball



print('libraries loaded...')



#load subj number and seed
#subj
ss = int(sys.argv[1])
#seed region
dorsal = str(sys.argv[2])

print(ss, dorsal)
# %%
#setup directories
study ='docnet'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/vayzenbe/GitHub_Repos/docnet/results'
exp = 'catmvpa'

sub_dir = f'{study_dir}/sub-{study}{ss}/ses-02/'
cov_dir = f'{sub_dir}/covs'
roi_dir = f'{sub_dir}/derivatives/rois'
exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'

runs = list(range(1,9))

whole_brain_mask = image.load_img('/user_data/vayzenbe/GitHub_Repos/fmri/roiParcels/mruczek_parcels/binary/all_visual_areas.nii.gz')
affine = whole_brain_mask.affine
dimsize = whole_brain_mask.header.get_zooms()  #get dimenisions

# scan parameters
vols = 331
first_fix = 8

# threshold for PCA
pc_thresh = .9

clf = LinearRegression()
#train/test split in 6 and 2 runs
rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)

"""
Setup searchlight
"""
print('Searchlight setup ...')
#set search light params

mask = image.get_data(whole_brain_mask) #the mask to search within


sl_rad = 2 #radius of searchlight sphere (in voxels)
max_blk_edge = 10 #how many blocks to send on each parallelized search
pool_size = 1 #number of cores to work on each search

voxels_proportion=1
shape = Ball

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
def calc_mvc(seed_train,seed_test, target_train, target_test, target_pc):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs

    seed_train,seed_test, target_train, target_test, target_pc
    """

    all_corrs = []
    for pcn in range(0,len(target_pc.explained_variance_ratio_)):
        
        clf.fit(seed_train, target_train[:,pcn]) #fit seed PCs to target
        pred_ts = clf.predict(seed_test) #use dorsal test data to predict left out runs of ventral test data
        weighted_corr = np.corrcoef(pred_ts,target_test[:,pcn])[0,1] * target_pc.explained_variance_ratio_[pcn]
        all_corrs.append(weighted_corr)

    final_corr = np.sum(all_corrs)/(np.sum(target_pc.explained_variance_ratio_))

    return final_corr


# %%

def create_ts_mask(train, test):
    """
    Create timeseries mask (i.e., a list of value)  that correspond to training and test runs
    """

    train_index = []
    test_index = []

    for tr in train:
        train_index = train_index + list(range((tr-1) * (vols-first_fix),((tr-1) * (vols-first_fix)) + (vols-first_fix)))

    for te in test:
        test_index = test_index + list(range((te-1) * (vols-first_fix),((te-1) * (vols-first_fix)) + (vols-first_fix)))

    return train_index, test_index


# %%
def mvpd(data, sl_mask, myrad, seed_ts):
    """
    Run multivaraite pattern dependance analysis
    """
    
    # Pull out the data
    data4D = data[0]
    data4D = np.transpose(data4D.reshape(-1, data[0].shape[3]))
    #print('mvpd', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)

    mvc_list = []
    
    #set up train/test split
    for train_runs, test_runs in rs.split(runs): 
        
        #determine train time points and test time points
        train_index, test_index = create_ts_mask(train_runs, test_runs)
        
        
        #split seed ts in train and test
        seed_train = seed_ts[train_index,:]
        seed_test = seed_ts[test_index,:]
        
        #split target region timeseries into train and test
        target_train = data4D[train_index, :]
        target_test = data4D[test_index, :]
        
        #extract PCs from seed and target
        n_comp = calc_pc_n(extract_pc(seed_train),pc_thresh) #determine number of PCs in train_data using threshold
        
        seed_pca = extract_pc(seed_train, n_comp) #conduct PCA one more time with that number of PCs
        #print(seed_pca.shape)

        seed_train_pcs = seed_pca.transform(seed_train) #transform train data in PCs
        seed_test_pcs = seed_pca.transform(seed_test) #transform test data into PCs 
        

        #extract PCs from seed and target
        n_comp = calc_pc_n(extract_pc(target_train),pc_thresh) #determine number of PCs in train_data using threshold
        target_pca = extract_pc(target_train, n_comp) #conduct PCA one more time with that number of PCs
        

        target_train_pcs = target_pca.transform(target_train) #transform train data in PCs
        target_test_pcs = target_pca.transform(target_test) #transform test data into PCs

        mvc_list.append(calc_mvc(seed_train_pcs, seed_test_pcs, target_train_pcs, target_test_pcs, target_pca))


    return np.mean(mvc_list)     



# %%
def load_data():
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
def extract_seed_ts(bold_vol):
    """
    extract all data from seed region
    """

    #load seed
    seed_roi = image.get_data(image.load_img(f'{roi_dir}/spheres/{dorsal}_sphere.nii.gz'))
    reshaped_roi = np.reshape(seed_roi, (91,109,91,1))
    masked_img = reshaped_roi*bold_vol

    #extract voxel resposnes from within mask
    seed_ts = masked_img.reshape(-1, bold_vol.shape[3]) #reshape into rows (voxels) x columns (time)
    seed_ts =seed_ts[~np.all(seed_ts == 0, axis=1)] #remove voxels that are 0 (masked out)
    seed_ts = np.transpose(seed_ts)

    print('Seed data extracted...')

    return seed_ts

# %%


bold_vol = load_data()
seed_ts = extract_seed_ts(bold_vol)

# %%
#run searchlight
t1 = time.time()
print("Begin Searchlight", print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024))
sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge, shape = shape) #setup the searchlight
print('Distribute', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.distribute([bold_vol], mask) #send the 4dimg and mask

print('Broadcast', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024)
sl.broadcast(seed_ts) #send the relevant analysis vars
print('Run', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)/1024, flush= True)
sl_result = sl.run_searchlight(mvpd, pool_size=pool_size)
print("End Searchlight\n", (time.time()-t1)/60)


# %%
sl_result = sl_result.astype('double')  # Convert the output into a precision format that can be used by other applications
sl_result[np.isnan(sl_result)] = 0  # Exchange nans with zero to ensure compatibility with other applications
sl_nii = nib.Nifti1Image(sl_result, affine)  # create the volume image
nib.save(sl_nii, f'{out_dir}/{study}{ss}_{dorsal}_mvpd.nii.gz')  # Save the volume

