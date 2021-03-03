from torch.utils.data import Dataset
from toolkit.loaders.loader_eth import load_eth
import numpy as np
import torch
import cv2

class OpTrajData(Dataset):
    def __init__(self,dataset='ETH'):
        super(OpTrajData,self).__init__()
        # self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        self.root='/home/faith/GitRepos/OpenTraj'
        if dataset=='ETH':
            self.H=np.loadtxt(self.root+'datasets/ETH/seq_eth/H.txt')
            self.dataset=load_eth(self.root+'/datasets/ETH/seq_eth/obsmat.txt')
            self.trajectory=self.dataset.get_trajectories()
            self.groups=self.trajectory.indices
            self.video=cv2.VideoCapture(self.root+'datasets/ETH/seq_eth/video.avi')
            # import pdb; pdb.set_trace()

    def __len__(self):
        return len(self.groups)

    def getImages(self,inds):
        frames=[]
        for i in inds:
            self.video.set(1,i) #1 is for CV_CAP_PROP_POS_FRAMES
            ret, f=self.video.read()
            frames.append(f)
        return frames

    def __getitem__(self, item):
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        frames=self.getImages(data['frame_id'].tolist())
        positions=data.filter(['pos_x','pos_y']).to_numpy()
        return [torch.FloatTensor(positions),torch.FloatTensor(frames)]


# x=OpTrajData()
# pos, fram=x.__getitem__(3)
# import pdb; pdb.set_trace()