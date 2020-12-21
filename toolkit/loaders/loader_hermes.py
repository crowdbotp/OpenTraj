# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import pandas as pd
import numpy as np
import sys
import os
from toolkit.core.trajdataset import TrajDataset


def load_bottleneck(path, **kwargs):
    traj_dataset = TrajDataset()

    csv_columns = ["agent_id", "frame_id", "pos_x", "pos_y", "pos_z"]
    # read from csv => fill traj table
    raw_dataset = pd.read_csv(path, sep=r"\s+", header=None, names=csv_columns)

    # convert from cm => meter
    raw_dataset["pos_x"] = raw_dataset["pos_x"] / 100.
    raw_dataset["pos_y"] = raw_dataset["pos_y"] / 100.

    traj_dataset.title = kwargs.get('title', "no_title")

    # copy columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 16)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    transform = np.array([[0, 1, 0],
                          [1, 0, 0],
                          [0, 0, 1]])
    traj_dataset.apply_transformation(transform, inplace=True)

    return traj_dataset