
# Author: Pat Zhang
# Email: bingqing.zhang.18@ucl.ac.uk
import sys
import os
import math
import numpy as np
from numpy import newaxis
import sympy as sp 
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
import itertools
import seaborn as sns
import pandas as pd
from copy import deepcopy
from toolkit.loaders.loader_eth import load_eth
from toolkit.core.trajdataset import TrajDataset
from toolkit.core.trajlet import split_trajectories

#calculate DCA, TTCA for each agent at time t
#find min ttc, dca, energy for each agent with respect to all other agent at time t
#then take min value in this trajlet

def DCA_MTX(x_4d):
    """:param x_4d = 4d sample (position|velocity)"""
    N = len(x_4d)
    tiled_x = np.tile(x_4d, (N, 1, 1))
    diff = tiled_x - tiled_x.transpose((1, 0, 2))
    D_4d = diff
    Dp = D_4d[:,:,:2]
    Dv = D_4d[:,:,2:]

    DOT_Dp_Dv = np.multiply(Dp[:,:,0], Dv[:,:,0]) + np.multiply(Dp[:,:,1], Dv[:,:,1])
    Dv_sq = np.multiply(Dv[:,:,0], Dv[:,:,0]) + np.multiply(Dv[:,:,1], Dv[:,:,1]) + 1E-6

    TTCA = np.array(-np.divide(DOT_Dp_Dv, Dv_sq))
    TTCA[TTCA < 0] = 0
    
    DCA = np.stack([Dp[:,:,0] + TTCA * Dv[:,:,0],
                    Dp[:,:,1] + TTCA * Dv[:,:,1]], axis=2)
    DCA = np.linalg.norm(DCA, axis=2)
    
    #tri_TTCA = TTCA[np.triu_indices(TTCA.shape[0],1)]
    #tri_DCA = DCA[np.triu_indices(DCA.shape[0],1)]
    
    return DCA, TTCA



def ttc(all_frames,name,trajlets):
    all_ttc = []
    Rp = 0.33 #assume pedestrians radius is 0.33
    new_frames = []
    for frame in all_frames:
        frame.reset_index(inplace=True)
        #if there is only one pedestrian at that time, or encounter invalid vel value
        if len(frame.index)<2 or frame['vel_x'].isnull().values.any() or frame['vel_y'].isnull().values.any():
            continue

        #calculate ttc for each pair
        x_4d = np.stack((frame.pos_x.values,frame.pos_y.values,frame.vel_x.values,frame.vel_y.values),axis=1)
        DCA,TTCA = DCA_MTX(x_4d)

        for i in range(len(TTCA)):
            #find out ttc of one agent
            ttc = [TTCA[i][j] for j in range(len(TTCA[i])) if DCA[i][j]<2*Rp and TTCA[i][j]>0] 
            #find out min ttc for one agent
            if len(ttc)>0:
                min_ttc = np.min(ttc)
                frame.loc[i,'ttc'] = min_ttc

            
            min_dca = np.min([j for j in DCA[i] if j>0])
            frame.loc[i,'dca'] = min_dca
     
        new_frames.append(frame)
    new_frames = pd.concat(new_frames)
    new_traj = TrajDataset()
    new_traj.data = new_frames
    trajs = new_traj.get_trajectories(label="pedestrian")
    trajlets[name] = split_trajectories(trajs, to_numpy=False)

    #average local density o each trajlet
    avg_traj_ttc=[]
    avg_traj_dca=[]
    for trajlet in trajlets[name]: 
        avg_traj_ttc.append(np.min(trajlet['ttc'].dropna())) #min of min
        avg_traj_dca.append(np.min(trajlet['dca'].dropna())) #min of min

    return avg_traj_ttc,avg_traj_dca
    

def energy(avg_ttc,upperbound,lowerbound):
    ttc = [i for i in avg_ttc if i<upperbound and i>lowerbound]
    tau0 = upperbound 
    k=1 
    #calculate collision energy
    E = []
    for tau in ttc:
        E.append((k/tau**2)*math.exp(-tau/tau0))

    E = np.array(E).astype("float")
    
    return E


def run(datasets, output_dir):

    #interaction range
    upperbound = 3
    thre = 3
    lowerbound = 0.2
    all_names = ['ETH-Univ','ETH-Hotel','UCY-Zara','UCY-Univ','SDD-Coupa','SDD-bookstore','SDD-deathCircle','GC','InD-1','InD-2','KITTI','LCas-Minerva','WildTrack','Edinburgh','BN-1d-w180','BN-2d-w160']
    #list(datasets.keys())

    datasets_ttc = []
    datasets_dca = []
    datasets_collision_energy = []
    # Get all datasets
    for name in all_names:
        dataset = datasets[name]
        print("reading",name)
        all_frames = dataset.get_frames()

        #calculate and store ttc
        all_trajs = dataset.get_trajectories("pedestrian")
       
        trajlets = {}
        #calculate and store ttc,dca,collision energy 
        avg_ttc,avg_dca = ttc(all_frames,name,trajlets)
       
        allttc_pd = pd.DataFrame(data=np.zeros((len(avg_ttc),2)),columns=['name','ttc'])
        allttc_pd.iloc[:,0] = [name for i in range(len(avg_ttc))] 
        allttc_pd.iloc[:,1]= avg_ttc
        datasets_ttc.append(avg_ttc)
        outputFile = output_dir+"/"+name+'_ttc.h5'
        fw = open(outputFile, 'wb')
        pickle.dump(allttc_pd, fw)
        fw.close()

        alldca_pd = pd.DataFrame(data=np.zeros((len(avg_dca),2)),columns=['name','dca'])
        alldca_pd.iloc[:,0] = [name for i in range(len(avg_dca))] 
        alldca_pd.iloc[:,1]= avg_dca
        datasets_dca.append(avg_dca)
        outputFile = output_dir+"/"+name+'_dca.h5'
        fw = open(outputFile, 'wb')
        pickle.dump(alldca_pd, fw)
        fw.close()
       
        all_E = energy(avg_ttc,upperbound,lowerbound)
        collision_energy = pd.DataFrame(data=np.zeros((len(all_E),2)),columns=['name','collision_energy'])
        collision_energy.iloc[:,0] = [name for i in range(len(all_E))] 
        collision_energy.iloc[:,1]=all_E 
        datasets_collision_energy.append(all_E)
        outputFile = output_dir+"/"+name+'_collisionEnergy.h5'
        fw = open(outputFile, 'wb')
        pickle.dump(collision_energy, fw)
        fw.close()

    
  
    # down-sample each group.
    ttc_d = normalize_samples_with_histogram(datasets_ttc,max_n_samples=500, n_bins=50)
    dca_d = normalize_samples_with_histogram(datasets_dca,max_n_samples=500, n_bins=50)
    E_d = normalize_samples_with_histogram(datasets_collision_energy, max_n_samples=500, n_bins=50)

       
    # put samples in a DataFrame (required for seaborn plots)
    df_ttc = pd.concat([pd.DataFrame({'title': all_names[ii],
                                            'ttc': ttc_d[ii],
                                            }) for ii in range(len(all_names))])
    df_dca = pd.concat([pd.DataFrame({'title': all_names[ii],
                                            'dca': dca_d[ii],
                                            }) for ii in range(len(all_names))])
    df_E = pd.concat([pd.DataFrame({'title': all_names[ii],
                                            'collision_energy': E_d[ii],
                                            }) for ii in range(len(all_names))])

    print("making plots ...")

   
   
    sns.set(style="whitegrid")
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12, 3),sharex=True)

    sns.swarmplot(y='ttc', x='title', data=df_ttc, size=1,ax=ax1)
    ax1.set_xlabel('')
    ax1.set_yticks([0,  4.0,  8.0, 12.0])
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.label.set_size(8)
    ax1.yaxis.set_tick_params(labelsize=8)

    sns.swarmplot(y='dca', x='title', data=df_dca, size=1,ax=ax2)
    ax2.set_xlabel('')
    ax2.set_yticks([0, 1.0, 2.0,3.0,4.0,5.0])
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.yaxis.label.set_size(8)
    ax2.yaxis.set_tick_params(labelsize=8)


    sns.swarmplot(y='collision_energy', x='title', data=df_E, size=1,ax=ax3)
    ax3.set_xlabel('')
    plt.xticks(rotation=-20)
    ax3.set_yticks([0,  4,  8,  12])
    ax3.xaxis.set_tick_params(labelsize=8)
    ax3.yaxis.label.set_size(8)
    ax3.yaxis.set_tick_params(labelsize=8)
    plt.subplots_adjust(hspace=0.1)
    fig.align_ylabels()
    plt.savefig(os.path.join(output_dir, 'collision.pdf'), dpi=500, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    from toolkit.test.load_all import get_datasets, all_dataset_names
    from toolkit.utils.histogram_sampler import histogram_sampler, normalize_samples_with_histogram

    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]

    datasets = get_datasets(opentraj_root, all_dataset_names)

    run(datasets, output_dir)
