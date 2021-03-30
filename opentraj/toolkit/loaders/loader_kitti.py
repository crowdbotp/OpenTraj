# Author: Pat
# Email: bingqing.zhang.18@ucl.ac.uk

#loader for KITTI, in global frame--origin is defined as the imu position at the first frame in that scene
import pandas as pd
import numpy as np
import glob
from toolkit.core.trajdataset import TrajDataset
import math
from math import cos, sin, tan, pi
from copy import deepcopy


def load_kitti(path, **kwargs):
    traj_dataset = TrajDataset()
    traj_dataset.title = "KITTI"
    track_files_list = sorted(glob.glob(path+"/label/*.txt"))
    calib_files_list = sorted(glob.glob(path+"/calib/*.txt"))
    imu_files_list =  sorted(glob.glob(path+"/oxts/*.txt"))
    
    #load track data, calibration data, IMU data from all scenes 
    track_rawData = loadTrack(track_files_list) #(left camera coordinate)
    calib_rawData = loadCalib(calib_files_list)
    imu_rawData = loadIMU(imu_files_list)
    
    #convert track data to world coordinate (imu coordinate in the first frame of that scene)
    track_world_pos = track_camToworld(track_rawData,calib_rawData,imu_rawData) 
   
    track_rawData = pd.concat(track_rawData)
    track_rawData.reset_index(inplace=True, drop=True)

    traj_dataset.data[["frame_id","agent_id","label","scene_id"]] = track_rawData[["frame","agent_id","type","scene"]]
    traj_dataset.data[["pos_x", "pos_y","pos_z"]]  = track_world_pos[["pos_x", "pos_y","pos_z"]] 
         
    # post-process. For KITTI, raw data do not include velocity, velocity info is postprocessed
    fps = kwargs.get('fps', 10)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


def loadTrack(files):
    csv_columns = ['frame', 'agent_id', 'type', 'truncated', 'occulded', 'alpha', 'bbox_l', 'bbox_t', 'bbox_r', 'bbox_b', 'dimensions_h','dimensions_w','dimensions_l', 'location_x','location_y','location_z', 'rotation_y', 'score','scene']
    track_raw_dataset=[]
    new_id = 0
    
    for file in files:
        data = pd.read_csv(file, sep=" ", header=None, names=csv_columns)
        #extract pedestrian trajectories only
        data = data[data['type']=='Pedestrian']
        data['scene'] = [files.index(file)]*len(data) 
        data.sort_values(by=['agent_id','frame'], inplace=True)
        
        #re-id
        copy_id = deepcopy(data["agent_id"]) #added
        uid=np.unique(copy_id)
        for oneid in uid:
            oneid_idx = [idx for idx, x in enumerate(copy_id) if x == oneid]
            for j in oneid_idx:
                data.iloc[j,1] = new_id
            new_id +=1
                
        track_raw_dataset.append(data)        

    return track_raw_dataset


def loadCalib(files):
    calib_raw_dataset=[]
    for file in files:
        calib = pd.read_csv(file,sep=" ",header=None) 
        calib_raw_dataset.append(calib.values[-2:,1:-2]) #only keep two transformation rows
    return calib_raw_dataset



def loadIMU(files):
    imu_raw_dataset = []
    for file in files:
        raw_IMU = pd.read_csv(file,sep=" ",header=None) 
        imu_raw_dataset.append(raw_IMU.values)
    return imu_raw_dataset




def convertOxtToPose(raw_IMU,frame,Tr_0_inv=[]):
    '''
    converts a list of oxts measurements into metric poses,
    starting at (0,0,0) meters, OXTS coordinates are defined as
    x = forward, y = right, z = down (see OXTS RT3000 user manual)
    afterwards, pose{i} contains the transformation which takes a
    3D point in the i'th frame and projects it into the oxts
    coordinates of the first frame.
    '''
    # compute scale from first lat value
    scale = math.cos(raw_IMU[0,0] * pi / 180.0) #first element in this row [lat]

    # init pose
    pose= []
    
    
  # if there is no data => no pose
    if (len(raw_IMU[frame,:]) == 0):
        pose.append([])
        return np.array(pose)

    
  # converts lat/lon coordinates to mercator coordinates using mercator scale
    er = 6378137
    mx = scale * raw_IMU[frame,1] * pi * er / 180
    my = scale * er * np.log(tan((90+raw_IMU[frame,0]) * pi / 360))
    # translation vector
    t = np.array([[mx,my,raw_IMU[frame,2]]]).reshape(3,1)

  # rotation matrix (OXTS RT3000 user manual)
    rx = raw_IMU[frame,3] # roll
    ry = raw_IMU[frame,4] # pitch
    rz = raw_IMU[frame,5] # heading 
    Rx = np.array([[1,0,0],[0,cos(rx),-sin(rx)],[0,sin(rx),cos(rx)]]) # base => nav  (level oxts => rotated oxts)
    Ry = np.array([[cos(ry),0,sin(ry)], [0,1,0], [-sin(ry),0,cos(ry)]])# % base => nav  (level oxts => rotated oxts)
    Rz = np.array([[cos(rz),-sin(rz),0], [sin(rz),cos(rz),0], [0,0,1]]) # % base => nav  (level oxts => rotated oxts)
    R  = np.matmul(Rz,np.matmul(Ry,Rx))

  # normalize translation and rotation (start at 0/0/0)
    if len(Tr_0_inv) == 0:
        Tr_0_inv = np.linalg.inv(np.r_[np.c_[R,t],np.array([[0,0,0,1]])])

  # add pose
    pose.append(np.matmul(Tr_0_inv,np.r_[np.c_[R,t],np.array([[0,0,0,1]])]))
    #shape: selected time frame x 4 x 4
    return np.array(pose),Tr_0_inv




def track_camToworld(track_rawData,calib_rawData,imu_rawData):
    track_pose_world_data = []
    for i in range(len(track_rawData)):
        #track, imu, calib data for each scene
        track_pos_cam = track_rawData[i][['frame','location_x','location_y','location_z']]
        raw_IMU = imu_rawData[i]
        calib = calib_rawData[i]
        #first convert it from cam coordinate to current IMU coordinate 
        #reconstruct it to 4*4 matrix
        Tr_velTocam = np.r_[np.array(calib[0],dtype='float').reshape(3,4),np.array([[0.0,0.0,0.0,1.0]],dtype='float')] #convert to 4*4
        Tr_imuTovel = np.r_[np.array(calib[1],dtype = 'float').reshape(3,4),np.array([[0.0,0.0,0.0,1.0]],dtype='float')] #convert to 4*4
        Tr_camTovel = np.linalg.inv(Tr_velTocam) #4*4
        Tr_velToimu = np.linalg.inv(Tr_imuTovel) #4*4
        
 
        #get transform matrix
        _,Tr_0_inv=convertOxtToPose(raw_IMU,0)
       
        
        for j in track_pos_cam.values: #row
            track_pose_imu = np.matmul(Tr_velToimu,np.matmul(Tr_camTovel,np.r_[j[1:],[1]]))
            #Transform from current IMU coordinate to the IMU coordinate in the first frame (world coordinate for this scene)
            Tr_imuToworld,_=convertOxtToPose(raw_IMU,int(j[0]),Tr_0_inv)
            track_pose_world_data.append(np.matmul(Tr_imuToworld,track_pose_imu).tolist()[0][:3])

    track_pos_world = pd.DataFrame(data=np.array(track_pose_world_data),columns=['pos_x','pos_y','pos_z'])
    return track_pos_world


if __name__ == "__main__":
    import os, sys
    opentraj_root = sys.argv[1]
    kitti_root = os.path.join(opentraj_root, 'datasets/KITTI/data')
    dataset = load_kitti(kitti_root, title='kitti', use_kalman=True, sampling_rate=1)
    trajs = dataset.get_trajectories()
    print(len(trajs))

