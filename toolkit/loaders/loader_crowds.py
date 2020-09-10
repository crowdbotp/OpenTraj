# Author: From TrajNet++
# Email: amiryan.j@gmail.com

from collections import namedtuple
import os
import numpy as np
import pandas as pd
import scipy.interpolate

from toolkit.core.trajdataset import TrajDataset

TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)
# SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])


class CrowdLoader:
    def __init__(self, homog=[]):
        if len(homog):
            self.homog = homog
        else:
            self.homog = np.eye(3)

    def to_world_coord(self, homog, loc):
        """Given H^-1 and world coordinates, returns (u, v) in image coordinates."""
       
        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(homog, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2]

    def crowds_interpolate_person(self, ped_id, person_xyf):
        ## Earlier
        # xs = np.array([x for x, _, _ in person_xyf]) / 720 * 12 # 0.0167
        # ys = np.array([y for _, y, _ in person_xyf]) / 576 * 12 # 0.0208

        ## Pixel-to-meter scale conversion according to
        ## https://github.com/agrimgupta92/sgan/issues/5
        # xs_ = np.array([x for x, _, _ in person_xyf]) * 0.0210
        # ys_ = np.array([y for _, y, _ in person_xyf]) * 0.0239
        
        xys = self.to_world_coord(self.homog, np.array([[x, y] for x, y, _ in person_xyf]))
        xs, ys = xys[:, 0], xys[:, 1]
        
        fs = np.array([f for _, _, f in person_xyf])

        kind = 'linear'
        #if len(fs) > 5:
        #    kind = 'cubic'

        x_fn = scipy.interpolate.interp1d(fs, xs, kind=kind)
        y_fn = scipy.interpolate.interp1d(fs, ys, kind=kind)

        frames = np.arange(min(fs) // 10 * 10 + 10, max(fs), 10)
     
        return [TrackRow(int(f), ped_id, x, y)
                for x, y, f in np.stack([x_fn(frames), y_fn(frames), frames]).T]

    def load(self, filename):
        with open(filename) as annot_file:
            whole_file = annot_file.read()

            pedestrians = []
            current_pedestrian = []
        

            for line in whole_file.split('\n'):
                if '- Num of control points' in line or \
                '- the number of splines' in line:
                    if current_pedestrian:
                        pedestrians.append(current_pedestrian)
                    current_pedestrian = []
                    continue

                # strip comments
                if ' - ' in line:
                    line = line[:line.find(' - ')]

                # tokenize
                entries = [e for e in line.split(' ') if e]
                if len(entries) != 4:
                    continue

                x, y, f, _ = entries

                current_pedestrian.append([float(x), float(y), int(f)])

            if current_pedestrian:
                pedestrians.append(current_pedestrian)
        return [row for i, p in enumerate(pedestrians) for row in self.crowds_interpolate_person(i, p)]


def load_crowds(path, **kwargs):
    """:param path: string, path to folder"""
    # pass the homography matrix as well

    homog_file = kwargs.get("homog_file", "")
    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    raw_dataset = pd.DataFrame()
    
    data = CrowdLoader(Homog).load(path)
    raw_dataset["frame_id"] = [data[i].frame for i in range(len(data))]
    raw_dataset["agent_id"] = [data[i].pedestrian for i in range(len(data))]
    raw_dataset["pos_x"] = [data[i].x for i in range(len(data))]
    raw_dataset["pos_y"] = [data[i].y for i in range(len(data))]
  
    traj_dataset = TrajDataset()

    traj_dataset.title = kwargs.get('title', "Crowds")
    # copy columns
    traj_dataset.data[["frame_id", "agent_id",  "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 25)

    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


# test
if __name__ == "__main__":
    import os, sys
    import matplotlib.pyplot as plt
    opentraj_root = sys.argv[1]

    zara_01_vsp = os.path.join(opentraj_root, 'datasets/UCY/zara01/crowds_zara01.vsp')
    zara_hmg_file = os.path.join(opentraj_root, 'datasets/UCY/zara01/H.txt')
    zara_01_ds = load_crowds(zara_01_vsp, use_kalman=False, homog_file=zara_hmg_file)
    trajs = zara_01_ds.get_trajectories()
    trajs = [g for _, g in trajs]
    samples = zara_01_ds.get_entries()
    plt.scatter(samples["pos_x"], samples["pos_y"])
    plt.show()
    zara_02_vsp = os.path.join(opentraj_root, 'datasets/UCY/zara02/crowds_zara02.vsp')
    zara_02_video = os.path.join(opentraj_root, 'datasets/UCY/zara02/video.avi')
    zara_02_ds = load_crowds(zara_02_vsp, homog_file=zara_hmg_file)

    zara_03_vsp = os.path.join(opentraj_root, 'datasets/UCY/zara03/crowds_zara03.vsp')
    zara_03_ds = load_crowds(zara_03_vsp, homog_file=zara_hmg_file)
