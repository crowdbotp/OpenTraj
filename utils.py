import numpy as np
import torch

def world2image(traj_w, H_inv):
    # import pdb; pdb.set_trace()
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2])
    return traj_uvz[:, :2].astype(int)

def computeGlobalGroups(x, numGrids, gridSize):
    # x is the vector of positions for each person (numPeople,2)
    # numGrids is (num of x grid squares, num of y grid squares) to split the space into
    # gridSize is (x grid dimension, y grid dimension)

    # import pdb; pdb.set_trace()
    #define variables
    xGrids = [gridSize[0] / numGrids[0] * i for i in range(1, numGrids[0] + 1)]
    yGrids = [gridSize[1] / numGrids[1] * i for i in range(1, numGrids[1] + 1)]
    numPeds=x.shape[-2]
    gridInds=[[] for x in range(numGrids[0]*numGrids[1])]

    # get the indexes of the people in each grid square
    for i in range(numPeds):
        if x[i][0]<xGrids[0]:
            if x[i][1]<yGrids[0]:
                gridInds[0].append(i)
            elif x[i][1]<yGrids[1]:
                gridInds[3].append(i)
            else:
                gridInds[6].append(i)
        elif x[i][0]<xGrids[1]:
            if x[i][1]<yGrids[0]:
                gridInds[1].append(i)
            elif x[i][1]<yGrids[1]:
                gridInds[4].append(i)
            else:
                gridInds[7].append(i)
        else:
            if x[i][1]<yGrids[0]:
                gridInds[2].append(i)
            elif x[i][1]<yGrids[1]:
                gridInds[5].append(i)
            else:
                gridInds[8].append(i)
    return gridInds

def processGroups(gblVec, features, mode='coords'):
    gblFeatures=[]
    for batch in range(len(features)):
        temp=[[] for x in range(len(features[batch]))]
        for grp in gblVec[batch]:
            if mode=='coords':
                for person in grp:
                    inds=[grp[i] for i in range(len(grp)) if grp[i]!=person]
                    if len(inds)>0:
                        temp[person] = torch.sum(features[batch][person]-features[batch][inds],0)
                    else:
                        temp[person] = features[batch][person]
            elif mode=='hidden':
                pass
            else:
                print('Unknown mode in processGroups (utils.py)')
                import pdb; pdb.set_trace()
        gblFeatures.append(torch.stack(temp))
    return gblFeatures