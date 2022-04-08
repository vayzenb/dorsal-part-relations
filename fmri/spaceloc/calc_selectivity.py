import sys
sys.path.insert(0, '/user_data/vayzenbe/GitHub_Repos/docnet/fmri')
import fmri_funcs
import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import itertools
import matplotlib.pyplot as plt
import warnings
import pdb
import warnings
from scipy import stats
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
file_suf = '_supp'



study ='spaceloc'

subj_list=["spaceloc1001", "spaceloc1002", "spaceloc1003", "spaceloc1004", "spaceloc1005", "spaceloc1006", "spaceloc1007",
"spaceloc1008" ,"spaceloc1009", "spaceloc1010", "spaceloc1011" ,"spaceloc1012", 
"spaceloc2013", "spaceloc2014", "spaceloc2015" , "spaceloc2016" , "spaceloc2017" , "spaceloc2018"]

#subj_list=["spaceloc2017" , "spaceloc2018"]

loc_suf = "_spaceloc" #runs to pull ROIs from
exp = ['spaceloc','depthloc','distloc','toolloc'] #experimental tasks
exp_cope=[[4,5], [4,5], [4,5], [6,7]] #copes to test in each ROI; each minus fix
#exp_cope=[[1,2], [1,2], [1,2], [1,2]] #copes to test in each roi; each minus their contrast (e.g., space -feature)

first_runs=[[2,4],[1,2],[1,2],[1,2]] #which first level runs to extract acts for
 #which first level runs to extract acts ford

bin_size=100
peak_vox=200
max_vox =2000

cond = [['space','feature'], ['3D',"2D"], ['distance', 'luminance'], ['tool','non_tool']]
cond_names = list(itertools.chain(*cond))
#cond = [['space_loc','feature_loc'], ['3D_loc',"2D_loc"], ['distance_loc', 'luminance_loc'], ['tool_loc','non_tool_loc']]

rois = ["LO_toolloc",  'PPC_spaceloc', 'APC_spaceloc']


bool_extract_data = True
bool_calc_act = True
bool_calc_mvpa = False


study_dir = f"/lab_data/behrmannlab/vlad/{study}"

def plot_vsf(out_dir, df, roi,cond, y_ax, save, out_name):
    df = df[cond]
    df.columns = cond
    ax = df.plot.line()
    ax.set_xlabel("Number of Voxels")
    ax.set_ylabel(y_ax)
    
    plt.title(roi)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.ioff()
    if save == True:
        plt.savefig(f'{out_dir}/results/figures/{roi}_{out_name}_VSF.png',bbox_inches='tight')
        plt.close()
        print("saved a figure",f'{out_dir}/results/figures/{roi}_{out_name}_VSF.png')


def plot_vsf_group(out_dir, df_vals,df_err, roi,cond, y_ax, save, out_name):
    df = df_vals[cond]
    df.columns = cond
    
    ax = df.plot.line()
    ax.set_xlabel("Number of Voxels")
    ax.set_ylabel(y_ax)
    
    plt.title(roi)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.ioff()
    if save == True:
        plt.savefig(f'{out_dir}/{roi}_{out_name}_VSF.png',bbox_inches='tight')
        plt.close()

def plot_bar(sub_dir, df, roi,cond,y_ax,save,out_name):
    df = df[cond]
    ax = df.plot.bar()
    ax.set_ylabel(f"Mean {y_ax} of {peak_vox} voxels")
    
    plt.title(roi)
    plt.tight_layout()
    plt.ioff()
    
    if save == True:
        plt.savefig(f'{sub_dir}/results/figures/{roi}_{out_name}_activation.png',bbox_inches='tight')
        plt.close()

def extract_acts():
    """
    Extracts activation within each ROI from the HighLevel then first level
    """
    for ss in subj_list:
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
        raw_dir = f'{sub_dir}/results/beta'
        
            
        os.makedirs(raw_dir, exist_ok = True) 
    
        for rr in rois:
            for lr in ['l','r']: #set left and right
                
                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    n = 0
                    for ecn, exp_cond in enumerate(exp): #loop through each experiment within each ROI localizer
                        '''
                        Extract acts from HighLevel
                        '''    
                        fmri_funcs.extract_data(sub_dir, raw_dir, roi, exp_cond, cond[ecn],exp_cope[ecn])

                        for run in first_runs[ecn]:
                            '''
                            Extract acts from FirstLevel
                            '''
                            for cn, cope_num in enumerate(exp_cope[ecn]):
                                cope_nifti = f'{sub_dir}/fsl/{exp_cond}/run-0{run}/1stLevel.feat/stats/zstat{cope_num}_reg.nii.gz'
                                out = f'{raw_dir}/{roi}_{cond[ecn][cn]}_{run}'
                                roi_nifti = f'{sub_dir}/rois/{roi}.nii.gz'
                                bash_cmd  = f'fslmeants -i {cope_nifti} -m {roi_nifti} -o {out}.txt --showall --transpose'
                                print(bash_cmd)
                                
                                
                                out = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
                else:
                    print(ss, roi)
                                
def calc_selectivity():
    '''
    Analyze univariate data
    '''
    roi_cond = ['spaceloc','toolloc']

    for ss in subj_list:
        print(ss, 'selectivity')
        sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
        raw_dir = f'{sub_dir}/results/beta'
        results_dir = f'{sub_dir}/results/beta_summary'
        os.makedirs(results_dir, exist_ok = True)
        os.makedirs(f'{sub_dir}/results/figures', exist_ok = True)
    
        for rr in rois:
            for lr in ['l','r']: #set left and right

                roi = f'{lr}{rr}' #set roi
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    loc_file = f'{sub_dir}/rois/data/{roi}.txt'

                    loc_df = pd.read_csv(loc_file, sep="  ", header=None, names = ["x", "y", "z", "loc"])
                    if loc_df.shape[0] >= peak_vox:
                        n = 0
                        for ecn, exp_cond in enumerate(exp): #loop through each experiment within each ROI localizer
                            '''
                            Analyze mean for each condition
                            '''
                            for cc in cond[ecn]: #loop through each condition of that localizer
                                curr_df = fmri_funcs.organize_data(sub_dir,raw_dir, roi, cc, 'dist')
                                if n == 0:
                                    df = curr_df
                                else:
                                    df[cc] = curr_df[cc]
                                n = n+1
                            
                        df.to_csv(f'{results_dir}/{ss}_{roi}_activations.csv', index = False)
                        
                        cond_name = list(itertools.chain(*cond))
                        df_sum = df.head(peak_vox)
                        df_sum = df_sum.mean()

                        
                        plot_bar(sub_dir, df_sum, roi,cond_name,'Beta', True,f'{roi}_activation')

                        df_roll = df.rolling(bin_size, win_type='triang').mean()
                        df_roll = df_roll.dropna()
                        df_roll= df_roll.reset_index(drop=True)
                        df_roll = df_roll.head(max_vox)

                        plot_vsf(sub_dir,df_roll,roi,cond_name, 'Beta',True, f'{roi}_activation')


                    

def group_summaries(data_type):
    

    os.makedirs(f'{study_dir}/derivatives/results/summaries', exist_ok = True)
    os.makedirs(f'{study_dir}/derivatives/results/figures', exist_ok = True)
    for rr in rois:
        for lr in ['l','r']: #set left and right

            roi = f'{lr}{rr}' #set roi
            all_subs = []
            vox_len = []
            for ss in subj_list:
                
                sub_dir = f'{study_dir}/sub-{ss}/ses-01/derivatives'
                results_dir = f'{sub_dir}/results/beta_summary'

                #check if roi exists, and, if it does that it has more than the minimum number of voxels
                if os.path.exists(f'{sub_dir}/rois/{roi}.nii.gz'):
                    loc_file = f'{sub_dir}/rois/data/{roi}.txt'
                    loc_df = pd.read_csv(loc_file, sep="  ", header=None, names = ["x", "y", "z", "loc"])
                    
                    if loc_df.shape[0] >= peak_vox:
                        curr_sub = pd.read_csv(f'{results_dir}/{ss}_{roi}_{data_type}.csv')
                        curr_sub = curr_sub[cond_names]
                        
                        vox_len.append(len(curr_sub)) #append the length of each df
                        all_subs.append(curr_sub.to_numpy())
            
            #shorten each df to the length of the smallest subj's dataframe so that they match
            max_vox = min(vox_len)
            for ssd in range(0, len(all_subs)):
                all_subs[ssd] = all_subs[ssd][0:max_vox,:]

            
            df_mean = pd.DataFrame(np.mean(np.array(all_subs), axis =0), columns=cond_names)
            df_se = pd.DataFrame(stats.sem(np.array(all_subs), axis =0), columns=cond_names)

            df_mean.to_csv(f'{study_dir}/derivatives/results/{roi}_mean_{data_type}{file_suf}.csv', index = False)
            df_se.to_csv(f'{study_dir}/derivatives/results/{roi}_se_{data_type}{file_suf}.csv', index = False)
            
            #plot_vsf(f'{study_dir}/derivatives/', df_mean,  roi, cond_names, data_type, True, data_type)
#            pdb.set_trace()
            
            #df_sum = df_mean.mean()
            
            #plot_bar(f'{study_dir}/derivatives/', df_sum, roi,cond_names, data_type, True,data_type)


                
        




subj_list = ["spaceloc2013", "spaceloc2014", "spaceloc2015" , "spaceloc2016" , "spaceloc2017" , "spaceloc2018"]
extract_acts()
calc_selectivity()



subj_list=["spaceloc1001", "spaceloc1002", "spaceloc1003", "spaceloc1004", "spaceloc1005", "spaceloc1006", "spaceloc1007",
"spaceloc1008" ,"spaceloc1009", "spaceloc1010", "spaceloc1011" ,"spaceloc1012", 
"spaceloc2013", "spaceloc2014", "spaceloc2015" , "spaceloc2016" , "spaceloc2017" , "spaceloc2018"]
group_summaries('activations')





