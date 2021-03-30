# Author: Pat Zhang
# Email: bingqing.zhang.18@ucl.ac.uk

import sys
import os
import math
import numpy as np
import pandas as pd
import sympy as sp 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from toolkit.loaders.loader_eth import load_eth
from toolkit.core.trajdataset import TrajDataset
from toolkit.core.trajlet import split_trajectories
from toolkit.test.load_all import get_datasets, all_dataset_names
from toolkit.utils.histogram_sampler import normalize_samples_with_histogram
from copy import deepcopy


def local_density(all_frames,trajlets,name):
    #define local density function
    #for all pedestrians at that time, find its distance to NN
    distNN = []
    dens_t = []
    a=1
    new_frames = []
    for frame in all_frames:
       
        if len(frame)>1:
            #find pairwise min distance
            distNN.append([])
            dens_t.append([])
            dist = squareform(pdist(frame[['pos_x','pos_y']].values))
            pair_dist = []
            for pi in dist:
                
                pair_dist.append(np.array(pi))
                min_pi = [j for j in pi if j>0.01]
                if len(min_pi) == 0:
                    min_dist = 0.01
                else:
                    min_dist = np.min(min_pi)
                distNN[-1].append(min_dist)

            #calculate local density for agent pj
            for pj in range(len(dist)):
                dens_t_i = 1/(2*np.pi)*np.sum(1/((a*np.array(distNN[-1]))**2)*np.exp(-np.divide((pair_dist[pj]**2),(2*(a*np.array(distNN[-1]))**2))))
                dens_t[-1].append(dens_t_i)
                frame.loc[frame.index[pj],'p_local'] = dens_t_i
        new_frames.append(frame)
    new_frames = pd.concat(new_frames)
    new_traj = TrajDataset()
    new_traj.data = new_frames

     
    trajs = new_traj.get_trajectories(label="pedestrian")
    trajlets[name] = split_trajectories(trajs, to_numpy=False)

    #average local density for each trajlet
    avg_traj_plocal=[]
    for trajlet in trajlets[name]:
        avg_traj_plocal.append(np.max(trajlet['p_local']))

               
    return avg_traj_plocal


def global_density(all_frames,area):
    #calculate global density as numebr of agents in the scene area at time t

    frame_density_samples = []
    new_frames = []
    for frame in all_frames:
        if len(frame)>0:
            oneArea = area.loc[frame['scene_id'].values[0],'area']
            frame_density_samples.append(len(frame) / oneArea)
            

    return frame_density_samples 


def run(datasets, output_dir):
    all_names = ['ETH-Univ','ETH-Hotel','UCY-Zara','UCY-Univ','SDD-Coupa','SDD-bookstore','SDD-deathCircle','GC','InD-1','InD-2','KITTI','LCas-Minerva','WildTrack','Edinburgh','BN-1d-w180','BN-2d-w160']

    #list(datasets.keys())

    #store all the results in pandas dataframe
    all_global_density=[]
    all_local_density=[]
    # Get trajectories from dataset
    for ds_name in all_names:
        dataset = datasets[ds_name]

        all_frames = dataset.get_frames()
        all_trajs = dataset.get_trajectories()

        trajlets = {}

        #get scene area
        scenes_maxX =  dataset.data.groupby(['scene_id'])['pos_x'].max() 
        scenes_minX =  dataset.data.groupby(['scene_id'])['pos_x'].min()
        scenes_maxY =  dataset.data.groupby(['scene_id'])['pos_y'].max()
        scenes_minY =  dataset.data.groupby(['scene_id'])['pos_y'].min()


        area=pd.DataFrame(data=[],columns=['scene_id','area'])
        for idx in scenes_maxX.index:
            x_range = scenes_maxX.loc[idx]-scenes_minX.loc[idx]
            y_range = scenes_maxY.loc[idx]-scenes_minY.loc[idx]
            area.loc[idx,'area'] = x_range*y_range
            

        #calculate and store global density

        global_dens = global_density(all_frames,area)
        g_density = pd.DataFrame(data=np.zeros((len(global_dens),2)),columns=['ds_name','global_density'])
        g_density.iloc[:,0] = [ds_name for i in range(len(global_dens))]
        g_density.iloc[:,1] = global_dens

        all_global_density.append(global_dens)
        outputFile1 = output_dir+"/"+ds_name+'_globalDens.h5'
        fw = open(outputFile1, 'wb')
        pickle.dump(g_density, fw)
        fw.close()

        #calculate and store local density

        trajlets = {}
        local_dens = local_density(all_frames,trajlets,ds_name)
        l_density = pd.DataFrame(data=[],columns=['ds_name','local_density'])

        l_density.iloc[:,1] = local_dens 
        l_density.iloc[:,0] = [ds_name for i in range(len(l_density.iloc[:,1]))]
        all_local_density.append(local_dens) 
        outputFile2 = output_dir+"/"+ds_name+'_localDens.h5'
        fw = open(outputFile2, 'wb')
        pickle.dump(l_density, fw)
        fw.close()


        print(ds_name," finish")
   
       
    # down-sample each group.
    # down-sample each group.
    gdens_d = normalize_samples_with_histogram(all_global_density[:-2], max_n_samples=800, n_bins=50,quantile_interval=[0.05, 0.98])
    ldens_d = normalize_samples_with_histogram(all_local_density[:-2],max_n_samples=800, n_bins=50,quantile_interval=[0.05, 0.95])
    BNgdens_d = normalize_samples_with_histogram(all_global_density[-2:], max_n_samples=800, n_bins=50,quantile_interval=[0.05, 0.98])
    BNldens_d = normalize_samples_with_histogram(all_local_density[-2:],max_n_samples=800, n_bins=50,quantile_interval=[0.05, 0.95])


    # put samples in a DataFrame (required for seaborn plots)
    df_gdens = pd.concat([pd.DataFrame({'title': all_names[ii],
                                            'global_density': gdens_d[ii],
                                            }) for ii in range(len(all_names[:-2]))])

    df_ldens = pd.concat([pd.DataFrame({'title': all_names[ii],
                                            'local_density': ldens_d[ii],
                                            }) for ii in range(len(all_names[:-2]))])

    BN_gdens = pd.concat([pd.DataFrame({'title': all_names[-ii-1],
                                            'global_density': BNgdens_d[ii],
                                            }) for ii in range(2)])

    BN_ldens = pd.concat([pd.DataFrame({'title': all_names[-ii-1],
                                            'local_density': BNldens_d[ii],
                                            }) for ii in range(2)])

    print("making plots ...")

    sns.set(style="whitegrid")
    fig,axs = plt.subplots(2,2,figsize=(12, 2),gridspec_kw={'width_ratios': [4, 1],'height_ratios': [1, 1]})


    sns.swarmplot(y='global_density', x='title', data=df_gdens, size=1,ax=axs[0,0])
    axs[0,0].set_ylim([0, 0.08])
    axs[0,0].set_yticks([0, 0.02,0.04,0.06,0.08])
    axs[0,0].set_xlabel('')
    axs[0,0].yaxis.label.set_size(8)
    axs[0,0].yaxis.set_tick_params(labelsize=8)


    sns.swarmplot(y='local_density', x='title', data=df_ldens, size=1,ax=axs[1,0])
    axs[1,0].set_ylim([0, 6])
    axs[1,0].set_yticks([0, 2.0,4.0,6.0])
    axs[1,0].yaxis.label.set_size(8)
    axs[1,0].xaxis.set_tick_params(labelsize=8)
    axs[1,0].set_xlabel('')
    axs[1,0].tick_params(axis='x', labelrotation=-20)
    axs[1,0].yaxis.set_tick_params(labelsize=8)


    sns.swarmplot(y='global_density', x='title', data=BN_gdens, size=1,ax=axs[0,1])
    axs[0,1].set_ylim([0, 1.5])
    axs[0,1].set_yticks([0, 0.5,1,1.5])
    axs[0,1].set_xlabel('')
    axs[0,1].set_ylabel('')
    axs[0,1].yaxis.set_tick_params(labelsize=8)


    sns.swarmplot(y='local_density', x='title', data=BN_ldens, size=1,ax=axs[1,1])
    axs[1,1].set_ylim([0, 6])
    axs[1,1].set_yticks([0, 2,4,6])
    axs[1,1].set_xlabel('')
    axs[1,1].set_ylabel('')
    axs[1,1].yaxis.set_tick_params(labelsize=8)


    axs[1,1].xaxis.set_tick_params(labelsize=8)
    plt.setp(axs[0,0].get_xticklabels(), visible=False)
    plt.setp(axs[0,1].get_xticklabels(), visible=False)

    fig.align_ylabels(axs[:, :])

    plt.xticks(rotation=-20)
    plt.subplots_adjust(hspace=0.18,wspace=0.12)
    plt.savefig(os.path.join(output_dir, 'density.pdf'), dpi=500, bbox_inches='tight')
    plt.show()
        

if __name__ == "__main__":
    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]
    datasets = get_datasets(opentraj_root, all_dataset_names)

    run(datasets, output_dir)

