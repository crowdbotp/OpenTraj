# ETH Walking Pedestrians Dataset (EWAP)
This dataset contains two experiments coined as ETH (left image) and Hotel (right image):
<p align='center'>
  <img src='./seq_eth/reference.png' width=320\>
  <img src='./seq_hotel/reference.png' width=300\>
</p>

* This dataset is extensively used in Human Trajectory Prediction literature.

## Annotations

Each directory contains a video together with the annotation and the obstacle map used. Please read the following for some details about the provided files. In each sequence directory there is a info.txt file with some sequence specific information.

#### OBSMAT
The actual annotation is stored in the obsmat.txt file. Each line has this format

```
[frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
```

however `pos_z` and `v_z` (direction perpendicular to the ground) are not used. The positions and velocities are in meters and are obtained with the homography matrix stored in H.txt .
Please note that we tried to avoid annotating those subjects that were at the border of the scene, as their behavior might have been influenced by the presence of other pedestrians/obstacles not in the field of view of the camera. We tried to be as consistent as possible in this regard. 

#### GROUPS
We tried to keep note of the people that seemed to walk in groups. These are listed in the file groups.txt . Each line contains a list of id, that are those that form a group. The id are the same as those in the obsmat.txt file 

#### DESTINATIONS
The assumed destinations for all the subjects walking in the scene are stored in the file destinations.txt . This is of course a simplifyiing assumption, but it seems to work fine for us.

#### OBSTACLES
the obstacles are reported in the map.png file. To bring the obstacle from image to world coordinates, the use of the homography matrix is necessary.

- **WARNING**: on 17/09/2009 the dataset have been modified, the frame number in the obsmat had a wrong offset (Thanks for corrections to Paul Scovanner)

## Homography
Homography matrics are provided by dataset creators in `H.txt` files. You can use them to project the world-coord positions (or trajectories) on the reference images (or videos).

Below is a sample python function to transform a trajectory from world coordinate system to image pixels:

```python
import os
import numpy as np

def world2image(traj_w, H_inv):    
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    traj_homog = np.hstack((traj_w, np.ones((traj_w.shape[0], 1)))).T  
    # to camera frame
    traj_cam = np.matmul(H_inv, traj_homog)  
    # to pixel coords
    traj_uvz = np.transpose(traj_cam/traj_cam[2]) 
    return traj_uvz[:, :2].astype(int)    

H = (np.loadtxt(os.path.join(OPENTRAJ_ROOT, "datasets/ETH/seq_eth/H.txt")))
H_inv = np.linalg.inv(H)
world2image({TRAJ}, H_inv)  # TRAJ: Tx2 numpy array
```

## Load Dataset with Toolkit
In order to the load the datasets, we provided the [`loader_eth.py`](https://github.com/crowdbotp/OpenTraj/blob/55ae4190d6507c1e6555bb7ca053e59666a2e177/toolkit/loaders/loader_eth.py#L9)

```python
import os
from toolkit.loaders.loader_eth import load_eth
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
traj_dataset = load_eth(os.path.join(OPENTRAJ_ROOT, "datasets/ETH/seq_eth/obsmat.txt"))
trajs = traj_dataset.get_trajectories()
```

## License
No license information is available with this dataset.

## Citation
```
@inproceedings{pellegrini2009you,
  title={You'll never walk alone: Modeling social behavior for multi-target tracking},
  author={Pellegrini, Stefano and Ess, Andreas and Schindler, Konrad and Van Gool, Luc},
  booktitle={2009 IEEE 12th International Conference on Computer Vision},
  pages={261--268},
  year={2009},
  organization={IEEE}
}
```
* Please write to stefpell@vision.ee.ethz.ch for any question or comment.
