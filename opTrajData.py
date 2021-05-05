from torch.utils.data import Dataset
from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, GaussianBlur
from utils import world2image

class OpTrajData(Dataset):
    def __init__(self,dataset='ETH',mode='by_frame', image=None):
        super(OpTrajData,self).__init__()
        # self.root='/Users/faith_johnson/GitRepos/OpenTraj/'
        self.root='/home/faith/GitRepos/OpenTraj/'
        self.mode=mode
        self.image=image
        self.transforms=Compose([GaussianBlur(5)])
        if dataset=='ETH':
            self.H = np.loadtxt(self.root + 'datasets/ETH/seq_eth/H.txt')
            self.H=np.linalg.inv(self.H)
            video_path = 'datasets/ETH/seq_eth/video.avi'
            self.dataset=load_eth(self.root+'/datasets/ETH/seq_eth/obsmat.txt')
        elif dataset=='UCY':
            print('WARNING: MASKS/IMAGES ARE NOT CURRENTLY SUPPORTED FOR THIS DATASET')
            # hard coding to zara01
            self.H = np.loadtxt(self.root + 'datasets/UCY/zara01/H.txt')
            self.dataset = load_crowds('datasets/UCY/zara01/annotation.vsp', use_kalman=False,homog_file='datasets/UCY/zara01/H.txt')
            video_path = 'datasets/UCY/zara01/video.avi'


        if self.mode=='by_human':
            self.trajectory=self.dataset.get_trajectories()
            self.groups=self.trajectory.indices
        self.video=cv2.VideoCapture(self.root+video_path)



    def __len__(self):
        if self.mode=='by_human':
            return len(self.groups)
        elif self.mode=='by_frame':
            return self.dataset.data['frame_id'].nunique()-1

    def getImages(self,inds):
        frames=[]
        for i in inds:
            self.video.set(1,i) #1 is for CV_CAP_PROP_POS_FRAMES
            ret, f=self.video.read()
            frames.append(f)
        return frames

    def __getitem__(self, item):
        if self.mode=='by_human':
            pos, pos_1, frame = self.getOneHumanTraj(item)
            return pos, pos_1, frame
        elif self.mode=='by_frame':
            peopleIDs, pos, pos_1, frame = self.getOneFrame(item)
            return peopleIDs, pos, pos_1, frame

    def getMasks(self,im,locs):
        # import pdb; pdb.set_trace()
        frames=[]
        for ind, i in enumerate(im):
            fram = np.zeros_like(i)
            for loc in locs[ind]:
                pix_loc=world2image(np.array([loc]),self.H)
                # print(pix_loc)
                cv2.circle(fram,tuple(pix_loc[0]),5,(255,255,255),-1)
            frames.append(self.transforms(torch.FloatTensor(fram)))
            # cv2.imshow('test',frame)
            # cv2.waitKey()
        return frames

    def getOneFrame(self,item):
        # import pdb; pdb.set_trace()
        frameID=[self.dataset.data['frame_id'].unique()[item]]
        if self.image is not None:
            frame=self.getImages(frameID)
        else:
            frame=[]
        people=self.dataset.get_frames(frameID)[0]
        targ_people=[]
        i=1
        while len(targ_people)==0:
            targ_people=self.dataset.get_frames([frameID[0]+i])[0]
            i+=1
        inds = targ_people.agent_id.isin(people.agent_id)
        targ_people=targ_people[inds]
        targ_locs=targ_people.filter(['pos_x','pos_y']).to_numpy()
        inds = people.agent_id.isin(targ_people.agent_id)
        people = people[inds]
        peopleIDs = people['agent_id'].tolist()
        locs = people.filter(['pos_x', 'pos_y']).to_numpy()
        # import pdb; pdb.set_trace()
        if self.image == 'mask':
            frame = self.getMasks(frame, np.expand_dims(locs, 0))
        return [peopleIDs, locs, targ_locs, frame]

    def getOneHumanTraj(self,item):
        group=list(self.groups.keys())[item]
        data=self.trajectory.get_group(group)
        if self.image is not None:
            frames=self.getImages(data['frame_id'].tolist())
        else:
            frames=[]
        positions = data.filter(['pos_x', 'pos_y']).to_numpy()
        if self.image=='mask':
            frames=self.getMasks(frames,positions.reshape(len(positions),1,2))
        # return [torch.FloatTensor(positions),torch.FloatTensor(frames)]
        return [positions[:-1,:], positions[1:,:], frames]

# x=OpTrajData(dataset='UCY',image='mask', mode='by_frame')
# d=DataLoader(x,batch_size=1,shuffle=False)
# pos, pos_1, fram=x.__getitem__(3)
# import pdb; pdb.set_trace()
# for pid, pos, targ, img in d:
#     try:
#         # import pdb; pdb.set_trace()
#         cv2.imshow('',img[0].numpy()[0])
#         cv2.waitKey(100)
#         # print(pos)
#         # print(targ)
#     except:
#         import pdb; pdb.set_trace()
# for f in fram:
#     import pdb; pdb.set_trace()
#     cv2.imshow('',f)
#     cv2.waitKey(1000)