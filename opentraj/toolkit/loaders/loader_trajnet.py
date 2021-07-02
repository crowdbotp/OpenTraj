# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import pandas as pd
from opentraj.toolkit.core.trajdataset import TrajDataset


def load_trajnet(path, **kwargs):
    traj_dataset = TrajDataset()
    traj_dataset.title = kwargs.get('title', "TrajNet")

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]

    # read from csv => fill traj
    raw_dataset = pd.read_csv(path, sep="\s+", header=None, names=csv_columns)
    raw_dataset.replace('?', np.nan, inplace=True)
    raw_dataset.dropna(inplace=True)    

    # FIXME: in the cases you load more than one file into a TrajDataset Object

    # rearrange columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]
    
    traj_dataset.data["scene_id"] = kwargs.get("scene_id", 0)
    traj_dataset.data["label"] = "pedestrian"

    # calculate velocities + perform some checks
    if 'stanford' in path:
        fps = 30
    elif 'crowd' in path or 'biwi' in path:
        fps = 16
    else:
        fps = 7
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset
