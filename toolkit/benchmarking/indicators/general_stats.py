# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import sys
import numpy as np
import pandas as pd
from toolkit.core.trajdataset import TrajDataset
import datetime

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
    # total_duration = dataset.data.groupby(["scene_di", "agent_id"]).max() -\
    #                  dataset.data.groupby(["scene_di", "agent_id"]).min()

    timestamps = dataset.data[dataset.data["label"]
                              == "pedestrian"].groupby(["scene_id", "agent_id"])["timestamp"]
    dur = sum(timestamps.max() - timestamps.min())
    return dur

    # scene_ids = pd.unique(dataset.data["scene_id"])
    # total_duration = 0
    # for scene_id in scene_ids:
    #     scene_i_data = dataset.data.loc[dataset.data["scene_id"] == scene_id]
    #     scene_agent_ids = pd.unique(scene_i_data["agent_id"])
    #     for agent_id in scene_agent_ids:
    #         agent_i = scene_i_data.loc[scene_i_data["agent_id"] == agent_id]
    #
    #         agent_start_time = agent_i["timestamp"].min()
    #         agent_end_time = agent_i["timestamp"].max()
    #         total_duration += (agent_end_time - agent_start_time)
    # return total_duration


def num_trajlets(dataset: TrajDataset, length=4.8, overlap=2):
    trajs = dataset.get_trajectories(label="pedestrian")
    trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=0.)
    non_static_trajlets = split_trajectories(trajs, length, overlap, static_filter_thresh=1.)
    return len(trajlets), len(non_static_trajlets)


def main():
    from toolkit.benchmarking.load_all_datasets import get_datasets, all_dataset_names
    opentraj_root = sys.argv[1]  # e.g. os.path.expanduser("~") + '/workspace2/OpenTraj'
    # output_dir = sys.argv[2]  # e.g. os.path.expanduser("~") + '/Dropbox/OpenTraj-paper/exp/ver-0.2'

    dataset_names = all_dataset_names
    # dataset_names = ['ETH-Univ']
    datasets = get_datasets(opentraj_root, dataset_names)
    # datasets = get_datasets(opentraj_root, all_dataset_names)

    for ds_name , ds in datasets.items():
        # n_scenes = num_scenes(ds)
        n_agents = num_pedestrians(ds)
        dur = dataset_duration(ds)
        trajs_dur = total_trajectory_duration(ds)
        n_trajlets, n_non_static_trajlets = num_trajlets(ds)

        dur_td = datetime.timedelta(0, int(round(dur)), 0)
        trajs_dur_td = datetime.timedelta(0, int(round(trajs_dur)), 0)
        print(ds_name, ':', n_agents, dur_td, trajs_dur_td)
        print('# trajlets =', n_trajlets, '% non-static trajlets =', int(n_non_static_trajlets/n_trajlets*100))
        print('*******************')


if __name__ == "__main__":
    main()
