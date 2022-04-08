import numpy as np
import glob as glob
import pandas as pd
import pdb

skel_folder= f'/home/vayzenbe/GitHub_Repos/docnet/stim/coords' 


def center_skel(skel):
    """
    Centers the skeleton by calculating the center of mass (CoM)
    and then aligning all point to 0,0

    skel - the skeleton coordinates
    """
    origin = np.array([0,0])

    #calc CoM
    com = np.array([np.mean(skel[:,0]), np.mean(skel[:,1])])

    #calculate how far points need to be translated to get to origin
    trans = origin - com
    #translate points to origin
    trans_skel = skel + trans

    return trans_skel

def rotate_skel(skel,deg):
    """
    Find the maximal alignment between skels by
    iteratively rotating them by a certain degree

    skel - the skeleton coordinates
    deg - how much to rotate by
    """

    return skel


skelfiles = sorted(glob.glob(f'{skel_folder}/*.csv'))

skel_rdm = pd.DataFrame(columns=['obj1', 'obj2', 'similarity'])

for sk1n, sk1 in enumerate(skelfiles):
    obj1 = sk1.replace(skel_folder + '/', '')
    obj1 = obj1[:-7]

    skel1 = np.loadtxt(sk1, delimiter = ',')
    skel1 = skel1[:,0:2]

    skel1 = center_skel(skel1)


    for sk1n, sk2 in enumerate(skelfiles[sk1n+1:]):
        obj2 = sk2.replace(skel_folder + '/', '')
        obj2 = obj2[:-4]

        skel2 = np.loadtxt(sk2, delimiter = ',')
        skel2 = skel2[:,0:2]

        skel2 = center_skel(skel2)

        #compare skeleton by calculating the distance between one point on skel1 
        # with the closest point on skel2
        skel_dist = []
        for ii in range(0,len(skel1)):
            curr_point = []
            
            for jj in range(0,len(skel2)):
                curr_point.append(np.linalg.norm(skel1[ii,:] - skel2[jj,:]))
                
            
            #grab the closest distance value and save it
            skel_dist.append(np.min(curr_point))
        
        #find average dist
        skel_dist1= np.mean(skel_dist)

        #Calculate distance from skel2 to skel1
        skel_dist = []
        for ii in range(0,len(skel2)):
            curr_point = []
            for jj in range(0,len(skel1)):
                curr_point.append(np.linalg.norm(skel2[ii,:] - skel1[jj,:]))
                
            
            #grab the closest distance value and save it
            skel_dist.append(np.min(curr_point))
        #find average dist
        skel_dist2= np.mean(skel_dist)

        #average the final distance values
        final_vals = pd.Series([obj1, obj2, np.mean([skel_dist1,skel_dist2])], index=skel_rdm.columns)
        skel_rdm = skel_rdm.append(final_vals, ignore_index=True)

#pdb.set_trace()
skel_rdm.to_csv('rdms/skel_rdm.csv', index = False)



            




