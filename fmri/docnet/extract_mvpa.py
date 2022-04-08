import sys
sys.path.insert(0, '/user_data/vayzenbe/GitHub_Repos/docnet/fmri')
import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import warnings
import matplotlib
import matplotlib.pyplot as plt
import fmri_funcs
import seaborn as sns
import pdb
from scipy import stats



'''
sub info
'''
study ='docnet'
study_dir = f"/lab_data/behrmannlab/vlad/{study}"
subj_list=["docnet2001", "docnet2002","docnet2003","docnet2004", "docnet2005", "docnet2007",
"docnet2008", "docnet2012","docnet2013", "docnet2014", "docnet2015", "docnet2016"]
subj_list=[ "docnet2017", "docnet2018"]

 #runs to pull ROIs from
rois = ["LO_toolloc", 'PPC_spaceloc', 'APC_spaceloc',  'APC_distloc', 'PPC_toolloc', 'APC_toolloc']


exp = 'catmvpa' #experimental tasks

file_suf = "_supp"
exp_suf = [""]


exp_cond = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5']
exp_cats = ['boat', 'camera',' car', 'guitar','lamp']
exp_cope=list(range(1,26))#copes for localizer runs; corresponding numerically t
#suf="_combined"
suf="_combined"

num_vox = 100

def copy_rois():
    """
    Copies ROIs from spaceloc directory to docnet one
    """
    print("copying rois...")
    spaceloc_dir = f"/lab_data/behrmannlab/vlad/spaceloc"
    
    for ss in subj_list:
        print(ss)
        space_rois = f'{spaceloc_dir}/sub-spaceloc20{ss[-2:]}/ses-01/derivatives/rois'
        sub_rois = f'{study_dir}/sub-{ss}/ses-02/derivatives/'
        bash_cmd = f'rsync -aP {space_rois} {sub_rois}'
        subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
        #shutil.copytree(space_rois, sub_rois)

def extract_acts():
    '''
    extract PEs for each condition
    '''
    print('extracting rois acts')
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                for es in exp_suf:
                    print(es)
                    
                    fmri_funcs.extract_data(sub_dir, raw_dir, f'{lr}{rr}', exp,exp_cond, exp_cope,es,'pe1')


def sort_by_functional():
    '''
    load each condition and sort by functional data
    '''
    print('sorting data by roi functional activation')
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    for es in exp_suf:
                        
                        n =0
                        for ec in exp_cond:
                            curr_cond = f'{ec}{es}' #combine condition with the suffix of the current highlevel 
                            #combine fROI data with exp data for each condition and sort it by voxel strength/distance
                            curr_df = fmri_funcs.organize_data(sub_dir,raw_dir, roi, curr_cond, 'dist')

                        
                        
                            if n == 0:
                                df = curr_df
                                df = df.rename(columns={f'{ec}{es}': ec})
                            else:
                                df[ec] = curr_df[f'{ec}{es}']
                            n = n+1
                    
                    
                        df = df.iloc[:,5:] #extract just the act columns
                        df = df.sub(df.mean(axis=1), axis=0) #mean center 
                                            
                        df.to_csv(f'{raw_dir}/{roi}_voxel_acts{es}.csv', index = False)


def create_individual_rdm():
    '''
    load voxel acts from each highlevel and correlate in pairs
    to make symmetric/asymmetric RDMs

    This func makes both combined rdms and odd/even splits
    
    Also calculate within/between for category, identity, between
    '''
    
    
    #run_pairs = [["_12",'_13', '_14'],['_34','_24', '_23' ]]
    if suf == '_split':
        run_pairs = [['_even'],['_odd']]
        print('Creating split asymmetric matrix for each sub..')
    elif suf == '_combined':
        run_pairs = [[''],['']]
        print('Creating combined symmetric matrix for each sub..')
    
    
    
    
    for ss in subj_list:
        summary_df =pd.DataFrame(columns = ['roi', 'identity', 'category','between'])
        
        sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
        raw_dir = f'{sub_dir}/results/beta/{exp}'
        results_dir = f'{sub_dir}/results/beta_summary/{exp}'
        os.makedirs(results_dir, exist_ok = True)
        os.makedirs(f'{results_dir}/figures', exist_ok = True)

        os.makedirs(raw_dir, exist_ok = True) 
        for lr in ['l','r']: #set left and right    
            for rr in rois:
                
                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    rdm_df = pd.DataFrame(columns= ['stim1', 'stim2', 'similarity'])

                    all_rdms =[]
                    for rpn, rp in enumerate(run_pairs[0]):
                        #load each datafile
                        df1 = pd.read_csv(f'{raw_dir}/{roi}_voxel_acts{run_pairs[0][rpn]}.csv')
                        df1 = df1.iloc[0:num_vox,:]
                        df2 = pd.read_csv(f'{raw_dir}/{roi}_voxel_acts{run_pairs[1][rpn]}.csv')
                        df2 = df2.iloc[0:num_vox,:]

                        rdm = np.zeros((len(exp_cond),len(exp_cond)))
                        
                        #correlate with other runs
                        #This will fill out the entire matrix
                        for c1n, c1 in enumerate(exp_cond):
                            for c2n, c2 in enumerate(exp_cond):
                                #correlate the condition from d1 with df2
                                rdm[c1n, c2n] = np.arctanh(np.corrcoef(df1[c1], df2[c2])[0,1])
                                #rdm[c1n, c2n] = np.linalg.norm(df1[c1] - df2[c2])
                                if rp == "" and c1n < c2n:
                                    rdm_df = rdm_df.append(pd.Series([c1, c2, np.corrcoef(df1[c1], df2[c2])[0,1]], index=rdm_df.columns), ignore_index=True)
                                
                        #append RDMs from each run pair
                        all_rdms.append(rdm)
                    #pdb.set_trace()
                    all_rdms = np.array(all_rdms)
                    #rdm_vec = np.array(rdm_vec)

                    if rp == "": #only save if its all run RSA
                        #Save vector version
                        
                        rdm_df.to_csv(f'{results_dir}/{roi}_rdm_vec.csv', index = False)

                    #average them together
                    comb_rdm = np.mean(all_rdms, axis =0)
                    np.savetxt(f'{results_dir}/{roi}_RDM{suf}.csv', comb_rdm, delimiter=',',fmt='%1.3f')

                    
                    
                    
                    #save plot
                    sns_plot  = sns.heatmap(comb_rdm, linewidth=0.5)
                    sns_plot.figure.savefig(f'{results_dir}/figures/{roi}_rdm{suf}.png')
                    plt.close()

                    #Pull out within-ident
                    ident_mat = np.identity(len(exp_cond))
                    ident_rdm = ident_mat * comb_rdm
                    ident_rdm[ident_rdm==0] = np.nan
                    ident_mean = np.nanmean(ident_rdm)
                    #ident_se = stats.sem(ident_rdm, nan_policy = 'omit')

                    #pull out between-cat
                    between_rdm = comb_rdm
                    np.fill_diagonal(between_rdm,0)
                    between_rdm[between_rdm==0] = np.nan
                    cat_means =[]
                    
                    #loop through category blocks
                    for ii in range(0,len(exp_cond), len(exp_cats)):
                        #extract corrs for a category
                        curr_cat = between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)]
                        #append the mean for that category
                        cat_means.append(np.nanmean(curr_cat))
                        #replace that category with nans so you can average later
                        between_rdm[ii:ii+len(exp_cats), ii:ii+len(exp_cats)] = np.nan

                    cat_mean = np.mean(cat_means)
                    #cat_se = stats.sem(cat_means, nan_policy = 'omit')
                    between_mean =  np.nanmean(between_rdm)
                    #between_se =   stats.sem(between_rdm, nan_policy = 'omit')
                    
                    summary_df = summary_df.append(pd.Series([roi, ident_mean, cat_mean, between_mean],index = summary_df.columns),ignore_index=True)
                    summary_df.to_csv(f'{results_dir}/mvpa_summary{suf}.csv', index = False)
                    print(ss, {lr},rr)


def create_combined_rdm():
    """
    Average RDM vectors from each participant into one mean RDM
    """
    print("Creating average RDM..")

    summary_df =pd.DataFrame(columns = ['stim1', 'stim2'])

    summary_dir = f'{study_dir}/derivatives/results/{exp}'
    os.makedirs(summary_dir, exist_ok = True)
    os.makedirs(f'{summary_dir}/figures', exist_ok = True)

    for lr in ['l','r']: #set left and right    
        for rr in rois:
            
            roi = f'{lr}{rr}' #set roi
        
            sub_n = 0
            for sn, ss in enumerate(subj_list):
                sub_dir = f'{study_dir}/sub-{ss}/ses-02/derivatives'
                results_dir = f'{sub_dir}/results/beta_summary/{exp}'

                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    sub_n = sub_n + 1
                    curr_df = pd.read_csv(f'{results_dir}/{roi}_rdm_vec.csv')

                    #pdb.set_trace()
                    if sn == 0:
                        all_df = curr_df
                    else:
                        all_df['similarity'] = all_df['similarity'] + curr_df['similarity']

            #print(roi, sub_n)
            summary_df['stim1'] = all_df['stim1']
            summary_df['stim2'] = all_df['stim2']  
            summary_df[roi] =1 - (all_df['similarity']/sub_n)
                    

    

    summary_df.to_csv(f'{summary_dir}/allrois_rdm{file_suf}.csv', index = False)
    





subj_list=[ "docnet2017", "docnet2018"]
copy_rois()

extract_acts()
sort_by_functional()

create_individual_rdm()


subj_list=["docnet2001", "docnet2002","docnet2003","docnet2004", "docnet2005", "docnet2007",
"docnet2008", "docnet2012","docnet2013", "docnet2014", "docnet2015", "docnet2016","docnet2017", "docnet2018"]

create_combined_rdm()






                





                    





