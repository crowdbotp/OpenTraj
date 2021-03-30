# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import sys
import numpy as np
import pandas as pd
import datetime

from toolkit.core.trajdataset import TrajDataset
from toolkit.core.trajlet import split_trajectories


def num_scenes(dataset: TrajDataset):
    n_scenes = dataset.data.groupby(["scene_id"]).count()
    return n_scenes


def dataset_duration(dataset: TrajDataset):
    timestamps = dataset.data.groupby(["scene_id"])["timestamp"]
    dur = sum(timestamps.max() - timestamps.min())
    return dur


def num_pedestrians(dataset: TrajDataset):
    return len(dataset.data[dataset.data["label"] == "pedestrian"].groupby(["scene_id", "agent_id"]))


def total_trajectory_duration(dataset: TrajDataset):
    timestamps = dataset.data[dataset.data["label"]
                              == "pedestrian"].groupby(["scene_id", "agent_id"])["timestamp"]
    dur = sum(timestamps.max() - timestamps.min())
    return dur


def num_trajlets(dataset: TrajDataset, length=4.8, overlap=2):
    trajs = dataset.get_trajectories(label="pedestrian")
    trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=0.)
    non_static_trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=1.)
    return len(trajlets), len(non_static_trajlets)


def run(datasets, output_dir):
    for ds_name, ds in datasets.items():
        # n_scenes = num_scenes(ds)
        n_agents = num_pedestrians(ds)
        dur = dataset_duration(ds)
        trajs_dur = total_trajectory_duration(ds)
        n_trajlets, n_non_static_trajlets = num_trajlets(ds)

        dur_td = datetime.timedelta(0, int(round(dur)), 0)
        trajs_dur_td = datetime.timedelta(0, int(round(trajs_dur)), 0)
        print(ds_name, ':', n_agents, dur_td, trajs_dur_td)
        print('# trajlets =', n_trajlets, '% non-static trajlets =', int(n_non_static_trajlets / n_trajlets * 100))
        print('*******************')


if __name__ == "__main__":
    from toolkit.test.load_all import get_datasets, all_dataset_names

    opentraj_root = sys.argv[1]
    dataset_names = all_dataset_names
    all_datasets = get_datasets(opentraj_root, dataset_names)
    run(all_datasets, '')
