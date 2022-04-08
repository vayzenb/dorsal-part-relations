import numpy as np
import pandas as pd
import subprocess
import os
import pdb


"""
assorted useful functions for analyzing fmri dataset
most functions use FSL commands 
"""


def extract_data(sub_dir, out_dir, roi, exp,cond_list, cope_list, suf='',stat_type='zstat1'):
    """
    Function to extract raw data from ROI and parameter estimate from each mask

    As input, takes:
    sub_dir, results_dir, roi, exp,cond_list, cope_list, stat_type

    """
    
    
    roi_nifti = f'{sub_dir}/rois/{roi}.nii.gz'
    
    if os.path.exists(roi_nifti):
           
        for ec in range(0,len(cope_list)):
            
            cope_nifti = f"{sub_dir}/fsl/{exp}/HighLevel{suf}.gfeat/cope{cope_list[ec]}.feat/stats/{stat_type}.nii.gz"
            out = f'{out_dir}/{roi}_{cond_list[ec]}{suf}'
           
            
            
            bash_cmd  = f'fslmeants -i {cope_nifti} -m {roi_nifti} -o {out}.txt --showall --transpose'
            print(bash_cmd)
            #pdb.set_trace()
            
            subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
            
            #print(bash_out.stdout)
    return 



def calc_distance(loc_df):
    """
    function to calculate the distance between the peak voxel and every other voxel in the localizer data
    
    As input, takes:
    a pandas dataframe
    
    """
    peak_vox = loc_df.iloc[0,0:3]

    all_coords = loc_df.iloc[:,0:3]

    dist = all_coords[['x', 'y', 'z']].sub(np.array(peak_vox)).pow(2).sum(1).pow(0.5)
    return dist


def organize_data(sub_dir,results_dir, roi, cond, sort_type):
    
    """
    Function to load and organize experimental data by localizer voxel strength (function or distance)
    
    Takes:
    sub_dir,results_dir, roi, cond, sort_type
    """

    #define and read localzier files
    loc_file = f'{sub_dir}/rois/data/{roi}.txt'

    loc_df = pd.read_csv(loc_file, sep="  ", header=None, names = ["x", "y", "z", "loc"])

    
    #define exp file
    exp_file = f'{results_dir}/{roi}_{cond}.txt'

    #load each file
    exp_df = pd.read_csv(exp_file, sep="  ", header=None, names = ["x", "y", "z", cond])

    #Append it to the localizer data
    loc_df = loc_df.join(exp_df[cond])


    #sort file by localizer functional value (high to low)
    loc_df = loc_df.sort_values(by =['loc'], ascending=False)
    loc_df = loc_df.reset_index(drop = True)
    if sort_type == 'dist':
        loc_df['dist'] = calc_distance(loc_df)

        loc_df = loc_df.sort_values(by =['dist', 'loc'], ascending=[True, False])
        loc_df= loc_df.reset_index(drop=True)

    loc_df = loc_df[["x", "y", "z", "loc", "dist", cond]]
    return loc_df

def calc_haxby_mvpa(roi_dir, results_dir, roi, cond_list,split_list, sort_type):
    """
    Function to do Haxby stlye MVPA
    """
    
    
    #define and read localzier files
    #Make sure this is an independant ROI
    loc_file = f'{results_dir}/{roi}.txt'

    loc_df = pd.read_csv(loc_file, sep="  ", header=None, names = ["x", "y", "z", "loc"])

    df = []
    for cc in cond_list:
        for sp in split_list:
            #define  odd and even exp file
            #note that it should be pulled from the opposite test runs (from even ROI pull odd data)
            #all naming convetions are relative to the ROI that data are being pulled
            #odd_exp_file is even data pulled from *odd* run ROI
            exp_file = f'{results_dir}/{roi}_{cc}.txt'
            exp_file = pd.read_csv(exp_file, sep="  ", header=None, names = ["x", "y", "z", f'{cc}_{sp}'])
            df = df + [exp_file]

            loc_df = loc_df.join([exp_file[f'{cc}_{sp}']])

    #sort file by localizer functional value (high to low)
    loc_df = loc_df.sort_values(by =['loc'], ascending=False)
    loc_df = loc_df.reset_index(drop = True)

    if sort_type == 'dist':
        loc_df = calc_distance(loc_df)

        loc_df = loc_df.sort_values(by =['dist', 'loc'], ascending=[True, False])
        loc_df= loc_df.reset_index(drop=True)
        
    #Leave onlyconditions of interest
    
    loc_df = loc_df.drop(columns["x", "y", "z", "loc", "dist"])

    #demean columns by condition
    #demeaning is important here, because you are correlating across voxel and so you might have 
    #a shadow correlation because that voxel is arbitrarily high
    row_mean=loc_df.mean(axis=1)
    
    #row_mean=loc_df.iloc[:,4:loc_df.shape[1]].mean(axis=1)
    #loc_df.iloc[:,4:loc_df.shape[1]] =loc_df.iloc[:,4:loc_df.shape[1]].sub(row_mean,axis=0)
    exp_df =loc_df.sub(row_mean,axis=0)

    #Start within-between analysis
    n = 1
    df = pd.DataFrame()
    between_temp = pd.DataFrame() 
    temp_df = []
    for c1 in cond_list:
        for c2 in cond_list:
            for sp in split_list:
                temp_df = temp_df + exp_df[f'{c1}_{sp}']

            temp_x = exp_df[f'{c1}_odd']
            temp_y = exp_df[f'{c2}_even']
            temp = temp_x.rolling(bin_size).corr(temp_y)
            temp = temp.dropna()
            temp= temp.reset_index(drop=True)

            if c1 == c2:
                temp = pd.DataFrame(temp)
                temp.columns= [f'{c1}']
                if df.empty:
                    df = temp
                else:
                    df = df.join(temp)
            else:
                if between_temp.empty:
                    between_temp =temp
                else:
                    between_temp =between_temp + temp
                    n = n + 1

    between = pd.DataFrame(between_temp/n)
    between.columns = ['between']
    df = df.join(between)


    #df = df.sub(between,axis=0)
    return df #,loc_df,exp_df, row_mean
