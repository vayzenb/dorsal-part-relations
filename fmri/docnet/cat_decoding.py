
'''
Decode a left-out object category from MVPA data
'''

import sys
from sklearn import svm
import numpy as np
import pdb
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd



subj_list=["docnet2001", "docnet2002","docnet2003","docnet2004", "docnet2005", "docnet2007",
"docnet2008", "docnet2012","docnet2013", "docnet2014", "docnet2015", "docnet2016", 'docnet2017', 'docnet2018']
#subj_list = ['docnet2017', 'docnet2018']

#anatomical ROI
d_roi = ['LOC','PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_distloc',  'APC_distloc', 'PPC_toolloc', 'APC_toolloc']
v_roi = ['LO_toolloc']

svm_test_size = .4
svm_splits = 20

exp_cond = [ 'boat_1', 'boat_2', 'boat_3', 'boat_4', 'boat_5',
'camera_1', 'camera_2', 'camera_3', 'camera_4', 'camera_5',
'car_1', 'car_2', 'car_3', 'car_4', 'car_5',
'guitar_1', 'guitar_2', 'guitar_3', 'guitar_4', 'guitar_5', 
'lamp_1', 'lamp_2', 'lamp_3', 'lamp_4', 'lamp_5']

exp_cats = ['boat', 'camera',' car', 'guitar','lamp']

#create a list of labels for classification
exp_labels = np.concatenate((np.ones((1,5)),np.ones((1,5))*2, np.ones((1,5))*3, np.ones((1,5))*4, np.ones((1,5))*5),axis =1)[0]
#pdb.set_trace()

data_dir = 'derivatives/results/beta/catmvpa'
file_suf = '_supp'

n_vox = 100

#iteratively combine LO and PFS with one of the dorsal ROIS (or on its own)

#do cross-val SVM seperately for each sub
#combine across subs

def decode_category():
    all_rois = ['LO_toolloc', 'PFS_toolloc', 'PPC_spaceloc', 'APC_spaceloc', 'PPC_depthloc', 'APC_depthloc', 'PPC_distloc',  'APC_distloc', 'PPC_toolloc', 'APC_toolloc']
    
    summary_df = pd.DataFrame(columns = ['sub'] + ["l" + s for s in all_rois] + ["r" + s for s in all_rois])
    for sn, ss in enumerate(subj_list):
        subj_dir = f'/lab_data/behrmannlab/vlad/docnet/sub-{ss}/ses-02/{data_dir}'
        roi_dir = f'/lab_data/behrmannlab/vlad/docnet/sub-{ss}/ses-02/derivatives/rois'

        roi_decode = []
        
        
        for lr in ['l','r']:
            for rr in all_rois:
                roi_data = np.zeros((len(exp_cond), n_vox))
              
                for ecn, ec in enumerate(exp_cond):
                    #load in ventral data
                    #print(f'{subj_dir}/{lr}{rr}_{ec}.txt')
                    if os.path.exists(f'{subj_dir}/{lr}{rr}_{ec}.txt'):
                        temp_data = np.loadtxt(f'{subj_dir}/{lr}{rr}_{ec}.txt')
                        if len(temp_data) >= n_vox:
                            roi_data[ecn,:] = temp_data[0:n_vox,3]
                        else:
                            roi_data[ecn,0:len(temp_data)] = temp_data[:,3]
                            
                        
                
                

                #check if ROI exists or is an LOC run before doing SVM
                if os.path.exists(f'{roi_dir}/{lr}{rr}.nii.gz'):
                    #print(f'{roi_dir}/{lr}{rr}.nii.gz')
                    #run SVM
                    X = roi_data
                    y = exp_labels
                    
                    sss = StratifiedShuffleSplit(n_splits=svm_splits, test_size=svm_test_size)
                    sss.get_n_splits(X, y)

                    roi_acc = []
                    for train_index, test_index in sss.split(X, y):

                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        #pdb.set_trace()
                        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                        clf.fit(X_train, y_train)   

                        roi_acc.append(clf.score(X_test, y_test))
                        #print(clf.score(X_test, y_test))

                    #append each ROI score to 
                    roi_decode.append(np.mean(roi_acc))
                    print(ss, f'{lr}{rr}', np.mean(roi_acc))
                else:# if roi doesn't exist, make it NAN
                    roi_decode.append(np.NaN)
                    #print(f'{roi_dir}/{lr}{rr}.nii.gz')
                  
        summary_df = summary_df.append(pd.Series([ss] + roi_decode, index = summary_df.columns), ignore_index= True)
        summary_df.to_csv(f'results/decoding_summary_single_roi{file_suf}.csv', index = False)
                





decode_category()

