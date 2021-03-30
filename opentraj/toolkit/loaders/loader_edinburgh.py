# Author: Pat
# Email:bingqing.zhang.18@ucl.ac.uk
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import ast
import sys
import cv2
from toolkit.core.trajdataset import TrajDataset
from copy import deepcopy

#tested with date: 01Aug, 01Jul, 01Sep

def get_homog():
    image_9points = [[155,86.6],[350,95.4],[539,106],[149,206],[345,215],[537,223],[144,327],[341,334],[533,340]]
    #use top left points as origin
    image_9points_shifted = [[x-image_9points[0][0],y-image_9points[0][1]] for x,y in image_9points] 
    #world plane points
    d_vert = 2.97
    d_hori = 4.85
    world_9points=[]
    for i in range(3):
        for j in range(3):
            world_9points.append([d_hori*j,d_vert*i])
        
    #find homog matrix   
    h, status = cv2.findHomography(np.array(image_9points_shifted), np.array(world_9points))

    return h


# loaded all Edinburgh tracks files together
def load_edinburgh(path, **kwargs):
    traj_dataset = TrajDataset()
    traj_dataset.title = "Edinburgh"

    if os.path.isdir(path):
        files_list = sorted(glob.glob(path + "/*.txt"))
    elif os.path.exists(path):
        files_list = [path]
    else:
        raise ValueError("loadEdinburgh: input file is invalid")

    csv_columns = ['centre_x','centre_y','frame','agent_id','length'] 

    # read from csv => fill traj table 
    raw_dataset = []
    scene = []
    last_scene_frame = 0
    new_id = 0
    scale = 0.0247
    # load data from all files
    for file in files_list:
        data = pd.read_csv(file, sep="\n|=", header=None,index_col=None)
        data.reset_index(inplace =True)
        properties = data[data['index'].str.startswith('Properties')]
        data = data[data['index'].str.startswith('TRACK')]
        
        #reconstruct the data in arrays 
        track_data = []
        print("reading:"+str(file))
        for row in range(len(data)):
            one_prop = properties.iloc[row,1].split(";")
            one_prop.pop()
            one_prop = [ast.literal_eval(i.replace(' ',',')) for i in one_prop]
            track_length = one_prop[0][0]
           
            one_track = data.iloc[row,1].split(";")
            one_track.pop()
            one_track[0] = one_track[0].replace('[[','[')
            one_track[-1] = one_track[-1].replace(']]',']')
            one_track =np.array([ast.literal_eval(i.replace(' [','[').replace(' ',',')) for i in one_track])
            one_track = np.c_[one_track, np.ones(one_track.shape[0],dtype=int)*row, track_length* np.ones(one_track.shape[0],dtype=int)]
            track_data.extend(one_track)

        #clear repeated trajectories
        track_data_pd = pd.DataFrame(data =np.array(track_data), columns=csv_columns)
      
        clean_track = []
        for i in tqdm(track_data_pd.groupby('agent_id')):
            i[1].drop_duplicates(subset ="frame", keep = 'first', inplace = True)
            # clean repeated trajectory for the same agent 
            
            for j in i[1].groupby(['frame','centre_x','centre_y']):
                j[1].drop_duplicates(subset ="frame", keep = 'first', inplace = True)
                clean_track.append(j[1])
        clean_track = np.concatenate(clean_track)
        
        
        #re-id
        uid=np.unique(clean_track[:,3])
        ##added!!
        copy_id = deepcopy(clean_track[:,3])
        
        for oneid in uid:
            oneid_idx = [idx for idx, x in enumerate(copy_id) if x == oneid] 
            for j in oneid_idx:
                clean_track[j,3] = new_id
            new_id +=1

        scene.extend([files_list.index(file)]*len(clean_track))
        
        raw_dataset.extend(clean_track.tolist())
    raw_dataset = pd.DataFrame(np.array(raw_dataset), columns=csv_columns)
    raw_dataset.reset_index(inplace=True, drop=True)
    
    #find homog matrix
    H = get_homog()
    #apply H matrix to the image point
    img_data = raw_dataset[["centre_x","centre_y"]].values
    world_data = []
    for row in img_data:
        augImg_data=np.c_[[row],np.array([1])]
        world_data.append(np.matmul(H,augImg_data.reshape(3,1)).tolist()[:2])
        
    raw_dataset["centre_x"] = np.array(world_data)[:,0]
    raw_dataset["centre_y"] = np.array(world_data)[:,1] 

    traj_dataset.data[["frame_id", "agent_id","pos_x", "pos_y"]] = raw_dataset[["frame", "agent_id","centre_x","centre_y"]]
    traj_dataset.data["scene_id"] = kwargs.get("scene_id", scene)

    traj_dataset.data["label"] = "pedestrian"

    traj_dataset.title = kwargs.get('title', "Edinburgh")

    # post-process. For Edinburgh, raw data do not include velocity, velocity info is postprocessed
    fps = kwargs.get('fps', 9)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    print("finish")
    return traj_dataset



if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt

    opentraj_root = sys.argv[1]
    edinburgh_dir = os.path.join(opentraj_root, 'datasets/Edinburgh/annotations')
    selected_days = ['01Sep', '02Sep', '04Sep', '05Sep', '06Sep', '10Sep']
    edinburgh_path = os.path.join(edinburgh_dir, 'tracks.%s.txt' % selected_days[0])
    traj_dataset = load_edinburgh(edinburgh_path, title="Edinburgh",
                                  use_kalman=False, scene_id=selected_days[0], sampling_rate=4)  # original framerate=9
    trajs = list(traj_dataset.get_trajectories())
    for traj in trajs:
        plt.plot(traj[1]["pos_x"], traj[1]["pos_y"])
    plt.title("L-CAS dataset")
    plt.show()
