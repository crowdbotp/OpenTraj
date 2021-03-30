# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from toolkit.core.trajdataset import TrajDataset
from toolkit.utils.histogram_sampler import normalize_samples_with_histogram


def path_length(trajectory: pd.DataFrame):
    traj_poss = trajectory[['pos_x', 'pos_y']].diff().dropna()
    travelled_distance = np.linalg.norm(traj_poss, axis=1).sum()
    return travelled_distance


def path_efficiency(trajectory: pd.DataFrame):
    """
     ratio of distance between the endpoints of a segment
     over actual length of the trajectory
    """
    actual_length = path_length(trajectory)
    end2end_dist = np.linalg.norm(np.diff(trajectory[['pos_x', 'pos_y']].iloc[[0, -1]], axis=0))
    return end2end_dist / actual_length


def path_efficiency_index(trajlets_np: np.ndarray):
    num = np.linalg.norm(trajlets_np[:, -1, :2] - trajlets_np[:, 0, :2], axis=1)
    denom = np.linalg.norm(np.diff(trajlets_np[:, :, :2], axis=1), axis=2).sum(axis=1)
    return num / denom

    # path_eff_samples = []
    # for trl in trajlets:
    #     path_eff_value = path_efficiency(trl)
    #     if not np.isnan(path_eff_value):
    #         path_eff_samples.append(path_eff_value)
    # return path_eff_samples


def run(trajlets, output_dir):
    path_eff_values = []
    dataset_names = list(trajlets.keys())

    for ds_name, ds in trajlets.items():
        path_eff_ind = path_efficiency_index(ds) * 100
        path_eff_values.append(path_eff_ind)
    path_eff_values = normalize_samples_with_histogram(path_eff_values, max_n_samples=800, n_bins=100,
                                                       quantile_interval=[0.1, 1])
    df_path_eff = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                           'path_eff': path_eff_values[ii],
                                           }) for ii in range(len(dataset_names))])

    print("making path eff plots ...")

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 1.6))

    ax1 = fig.add_subplot(111)
    sns.swarmplot(y='path_eff', x='title', data=df_path_eff, size=1)
    plt.ylim([90, 100])
    plt.xlabel('')
    # plt.xticks([])
    # ax1.set_yticks([0, 0.5, 1, 1.5, 2.])
    plt.ylabel('Path Efficiency (%)')
    plt.xticks(rotation=-20)
    ax1.yaxis.label.set_size(9)
    ax1.xaxis.set_tick_params(labelsize=8)
    ax1.yaxis.set_tick_params(labelsize=8)

    plt.savefig(os.path.join(output_dir, 'path_eff.pdf'), dpi=500, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    import sys
    from toolkit.test.load_all import all_dataset_names, get_trajlets

    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]

    # dataset_names = ['ETH-Univ', 'ETH-Hotel']
    dataset_names = all_dataset_names
    trajlets = get_trajlets(opentraj_root, dataset_names)

    run(trajlets, output_dir)
