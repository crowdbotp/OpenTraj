from torch.utils.data import Dataset
from toolkit.loaders.loader_eth import load_eth
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)

class OpTrajData(Dataset):
    def __init__(self,dataset='ETH',mode='by_frame'):
        super(OpTrajData,self).__init__()
        self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        # self.root='/home/faith/GitRepos/OpenTraj'
        self.mode=mode
        if dataset=='ETH':
            self.H=np.loadtxt(self.root+'datasets/ETH/seq_eth/H.txt')
            self.dataset=load_eth(self.root+'/datasets/ETH/seq_eth/obsmat.txt')
            if self.mode=='by_human':
                self.trajectory=self.dataset.get_trajectories()
                self.groups=self.trajectory.indices
            self.video=cv2.VideoCapture(self.root+'datasets/ETH/seq_eth/video.avi')
            # import pdb; pdb.set_trace()

    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode=='by_frame':
            return self.dataset.data['frame_id'].nunique()

    def getImages(self,inds):
        frames=[]
        for i in inds:
            self.video.set(1,i) #1 is for CV_CAP_PROP_POS_FRAMES
            ret, f=self.video.read()
            frames.append(f)
        return frames

    def __getitem__(self, item):
        if self.mode=='by_human':
            pos, frame = self.getOneHumanTraj(item)
        elif self.mode=='by_frame':
            pos, frame = self.getOneFrame(item)
        return pos,frame

    def getOneFrame(self,item):
        frameID=[self.dataset.data['frame_id'].unique()[item]]
        im=self.getImages(frameID)[0]
        people=self.dataset.get_frames(frameID)[0]
        frame=np.zeros_like(im)
        locs=people.filter(['pos_x','pos_y']).to_numpy()
        # import pdb; pdb.set_trace()
        for loc in locs:
            pix_loc=world2image(np.array([loc]),np.linalg.inv(self.H))
            cv2.circle(frame,tuple(pix_loc[0]),5,(255,255,255),-1)
        # cv2.imshow('test',frame)
        # cv2.waitKey()
        # return [torch.FloatTensor(locs), torch.FloatTensor(frame)]
        return [locs, frame]

    def getOneHumanTraj(self,item):
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        frames=self.getImages(data['frame_id'].tolist())
        positions=data.filter(['pos_x','pos_y']).to_numpy()
        # return [torch.FloatTensor(positions),torch.FloatTensor(frames)]
        return [positions, frames]

# x=OpTrajData()
# d=DataLoader(x,batch_size=1,shuffle=False)
# pos, fram=x.__getitem__(3)
# import pdb; pdb.set_trace()
# for pos, img in d:
    # import pdb; pdb.set_trace()
    # cv2.imshow('',img[0].numpy())
    # cv2.waitKey(1)