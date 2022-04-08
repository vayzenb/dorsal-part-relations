import os
import pandas as pd
from nilearn import image, plotting, input_data, glm
#from nilearn.glm import threshold_stats_img
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
import matplotlib.pyplot as plt
import pdb

import warnings

warnings.filterwarnings('ignore')
#TS need to be standardized first 


#subs = list(range(1001,1006))
#print(subs)
study ='spaceloc'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
out_dir = f'{study_dir}/derivatives/fc'
results_dir = '/user_data/vayzenbe/GitHub_Repos/docnet/results'
exp = 'spaceloc'
rois = ['PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_toolloc', 'APC_toolloc', 'PPC_distloc', 'APC_distloc']
control_tasks = ['distloc','toolloc','depthloc']
file_suf  = ''
'''scan params'''
tr = 1
vols = 321

whole_brain_mask = load_mni152_brain_mask()
mni = load_mni152_template()
brain_masker = input_data.NiftiMasker(whole_brain_mask,
    smoothing_fwhm=0, standardize=True)

'''run info'''
run_num =6
runs = list(range(1,run_num+1))
run_combos = []
#determine the number of left out run combos
for rn1 in range(1,run_num+1):
    for rn2 in range(rn1+1,run_num+1):
        run_combos.append([rn1,rn2])

#run_combos = run_combos[0:2]

def add_lo_coords(subs):
    """
    Add LO to the ROI coords file
    """
    print('extracting LOC coords...')
    parcels = ['LO']

    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl'
        parcel_dir = f'{roi_dir}/parcels'
        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
        roi_coords = roi_coords[roi_coords['roi'] != 'lLO']
        roi_coords = roi_coords[roi_coords['roi'] != 'rLO']
        
        
        
        for lr in ['l','r']:
            for pr in parcels:
                #load parcel
                roi = image.load_img(f'{parcel_dir}/{lr}{pr}.nii.gz')
                roi = image.math_img('img > 0', img=roi)

                control_zstat = image.load_img(f'{exp_dir}/toolloc/HighLevel_roi.gfeat/cope5.feat/stats/zstat1.nii.gz')
                coords = plotting.find_xyz_cut_coords(control_zstat,mask_img=roi, activation_threshold = .99)

                for rn in range(0,15):
                    curr_coords = pd.Series([rn, 'toolloc', f'{lr}{pr}'] + coords, index=roi_coords.columns)
                    roi_coords = roi_coords.append(curr_coords,ignore_index = True)

        roi_coords.to_csv(f'{roi_dir}/spheres/sphere_coords.csv', index=False)

def extract_roi_sphere(img, coords):
    roi_masker = input_data.NiftiSpheresMasker([tuple(coords)], radius = 6)
    seed_time_series = roi_masker.fit_transform(img)
    
    phys = np.mean(seed_time_series, axis= 1)
    
    phys = phys.reshape((phys.shape[0],1))
    
    return phys


def make_psy_cov(runs,ss, cond):
    sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
    cov_dir = f'{sub_dir}/covs'
    times = np.arange(0, vols*len(runs), tr)
    full_cov = pd.DataFrame(columns = ['onset','duration', 'value'])
    for rn, run in enumerate(runs):    
        
        curr_cov = pd.read_csv(f'{cov_dir}/SpaceLoc_{study}{ss}_Run{run}_{cond}.txt', sep = '\t', header = None, names = ['onset','duration', 'value'])
        #contrasting (neg) cov
        curr_cov['onset'] = curr_cov['onset'] + (vols*rn)
        full_cov = full_cov.append(curr_cov)
        #add number of vols to the timing cols based on what run you are on
        #e.g., for run 1, add 0, for run 2, add 321
        #curr_cov['onset'] = curr_cov['onset'] + ((rn_n)*vols) 
        #pdb.set_trace()
        
        #append to concatenated cov
    full_cov = full_cov.sort_values(by =['onset'])
    cov = full_cov.to_numpy()
    cov = cov.astype(float)
    #convolve to hrf
    psy, name = glm.first_level.compute_regressor(cov.T, 'spm', times)
    psy[psy>0] = 1
    psy[psy<=0] = 0

    return psy

def extract_cond_ts(ts, cov):
    """
    Extracts timeseries corresponding to blocks in a cov file
    """

    block_ind = (cov==1)
    block_ind = np.insert(block_ind, 0, True)
    block_ind = np.delete(block_ind, len(block_ind)-1)
    block_ind = (cov == 1).reshape((len(cov))) | block_ind

    new_ts = ts[block_ind]


    return new_ts
#runs = [1]

#ss = 1001
#rr = 'rPPC_spaceloc'
def conduct_gca():

    print('Running GCA...')
    tasks = ['spaceloc','distloc']
    cond = ['SA','FT']
    
    d_rois = ['lPPC','rPPC','lAPC','rAPC']
    v_rois = ['lLO','rLO']
    for ss in subs:
        sub_summary =pd.DataFrame(columns = ['sub','fold','task','condition','origin','target', 'f_diff'])
        
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
        cov_dir = f'{sub_dir}/covs'
        roi_dir = f'{sub_dir}/derivatives/rois'
        exp_dir = f'{sub_dir}/derivatives/fsl/{exp}'
        os.makedirs(f'{sub_dir}/derivatives/results/beta_ts', exist_ok=True)

        roi_coords = pd.read_csv(f'{roi_dir}/spheres/sphere_coords.csv')
        for rcn, rc in enumerate(run_combos): #determine which runs to use for creating ROIs
            
            #Extract timeseries from each run
            filtered_list = []
            for rn in rc:
                
                curr_run = image.load_img(f'{exp_dir}/run-0{rn}/1stLevel.feat/filtered_func_data_reg.nii.gz')
                curr_run = image.clean_img(curr_run,standardize=True)
                filtered_list.append(curr_run)

            #concat runs
            img4d = image.concat_imgs(filtered_list)

            print(ss,rcn,'loaded')

            for tsk in tasks:
                for drr in d_rois:
                    
                    #load peak voxel in dorsal roi
                    dorsal_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] ==tsk) & (roi_coords['roi'] ==drr)]
                    #Extract TS from dorsal roi
                    dorsal_ts = extract_roi_sphere(img4d,dorsal_coords[['x','y','z']].values.tolist()[0])
                    
                
                    for cc in cond:
                        #load behavioral data
                        #time adjusted using HRF to pull out boxcar
                        psy = make_psy_cov(rc, ss,cc)

                        #create dorsal ts for just that condition
                        dorsal_phys = extract_cond_ts(dorsal_ts, psy)
                        
                        for vrr in v_rois:
                            
                            ventral_coords = roi_coords[(roi_coords['index'] == rcn) & (roi_coords['task'] =='toolloc') & (roi_coords['roi'] ==vrr)]
                            #pdb.set_trace()
                            ventral_ts = extract_roi_sphere(img4d,ventral_coords[['x','y','z']].values.tolist()[0])
                            ventral_phys = extract_cond_ts(ventral_ts, psy)                            

                            #Add TSs to a dataframe to prep for gca
                            neural_ts= pd.DataFrame(columns = ['dorsal', 'ventral'])
                            neural_ts['dorsal'] = np.squeeze(dorsal_phys)
                            neural_ts['ventral'] = np.squeeze(ventral_phys)
                            
                            #calculate dorsal GCA F-test
                            gc_res_dorsal = grangercausalitytests(neural_ts[['ventral','dorsal']], 1, verbose=False)
                            
                            #calculate ventral GCA F-test
                            gc_res_ventral = grangercausalitytests(neural_ts[['dorsal','ventral']], 1,verbose=False)

                            #calc difference
                            f_diff = gc_res_dorsal[1][0]['ssr_ftest'][0]-gc_res_ventral[1][0]['ssr_ftest'][0]

                            curr_data = pd.Series([ss, rcn,tsk, cc, drr, vrr, f_diff], index=sub_summary.columns)
                            
                            
                            sub_summary = sub_summary.append(curr_data,ignore_index=True)
                            print(ss, tsk,cc, drr,vrr)

                        
        print('done GCA for', ss)                
        sub_summary.to_csv(f'{sub_dir}/derivatives/results/beta_ts/gca_summary.csv',index=False)


def summarize_gca():
    """
    Compile subject data into one summary
    """
    print('Creating summary across subjects...')
    
    
    df_summary = pd.DataFrame()
    tasks = ['spaceloc', 'distloc']
    cond = ['SA','FT']

    d_rois = ['lPPC','rPPC','lAPC','rAPC']
    #d_rois = ['rPPC','rAPC']
    v_rois = ['lLO','rLO']
    print(subs)
    for ss in subs:
        sub_dir = f'{study_dir}/sub-{study}{ss}/ses-01/'
        data_dir = f'{sub_dir}/derivatives/results/beta_ts'

        curr_df = pd.read_csv(f'{data_dir}/gca_summary.csv')
        #pdb.set_trace()
        curr_df = curr_df.groupby(['task','condition', 'origin','target']).mean()
        curr_data = [ss]
        col_index = ['sub']
        for tsk in tasks:
            for cc in cond:
                for drr in d_rois:
                    for vrr in v_rois:

                        #for dorsal origin
                        col_index.append(f'{tsk}_{cc}_{drr}_{vrr}')
                        curr_data.append(curr_df['f_diff'][tsk,cc, drr, vrr])
                        

        if ss == 1001:
            df_summary = pd.DataFrame(columns=col_index)
            df_summary = df_summary.append(pd.Series(curr_data,index = col_index),ignore_index=True)
        else:
            df_summary = df_summary.append(pd.Series(curr_data,index = col_index),ignore_index=True)

    df_summary.to_csv(f"{results_dir}/gca/all_roi_summary{file_suf}.csv", index = False)

#subs = list(range(2013,2019))
#subs = list(range(2018,2017,-1))
#add_lo_coords(subs)
#conduct_gca()
subs = list(range(1001,1013)) + list(range(2013,2019))

summarize_gca()





        







