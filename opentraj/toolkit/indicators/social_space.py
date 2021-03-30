# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import sys
import pandas as pd
import numpy as np
from sklearn.metrics import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt

from toolkit.test.load_all import all_dataset_names, get_datasets, get_trajlets
from toolkit.utils.histogram_sampler import histogram_sampler


def social_space(dataset: pd.DataFrame):
    """The minimum distance that each of agent see during their lifetime"""

    def frame_dist(group):
        # get all positions at t = frame_id
        X_t = group[["pos_x", "pos_y"]]  # .loc[indices_t]
        N_t = len(X_t)  # number of agents at t = frame_id
        if N_t > 1:
            # compute distance matrix between all pairs of agents
            DD_t = euclidean_distances(X_t)

            # remove the diagonal elements (or self-distances)
            DD_t = DD_t[~np.eye(N_t, dtype=bool)].reshape(N_t, N_t - 1)

            # calc min dist for each agent at t = frame_id
            minD_t = np.min(DD_t, axis=1)
            group["min_dist"] = minD_t
        else:
            group["min_dist"] = 1000

        return group

    # with calculated min dist
    dataset.reset_index(inplace=True)
    new_data = dataset.groupby(["scene_id", "frame_id"]).apply(frame_dist)

    # grouped = new_data.groupby(["scene_id", "agent_id"])
    # sspace = grouped["min_dist"].min()
    # sspace = sspace.to_numpy()

    trajlet_length = 4.8
    trajlet_overlap = 2.
    sspace = []
    def trajlet_min_social_space(group):
        n_frames = len(group)
        if n_frames < 2: return
        ts = group["timestamp"].to_numpy()
        dt = ts[1] - ts[0]
        eps = 1E-2
        f_per_traj = int(np.ceil((trajlet_length - eps) / dt))
        f_step = int(np.ceil((trajlet_length - trajlet_overlap - eps) / dt))
        md = group["min_dist"].to_numpy()
        # md[::f_step]

        for start_f in range(0, n_frames - f_per_traj, f_step):
            sspace.append(min(md[start_f:start_f + f_per_traj]))

        return group

    new_data.groupby(["scene_id", "agent_id"]).apply(trajlet_min_social_space)
    sspace = np.array(sspace)
    return sspace[sspace < 8]


if __name__ == "__main__":
    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]  # os.path.expanduser("~") + '/Dropbox/OpenTraj-paper/exp/ver-0.2'
    soc_space_values = []

    # dataset_names = all_dataset_names
    dataset_names = ['ETH-Univ', 'ETH-Hotel', 'SDD-coupa']   # test

    datasets = get_datasets(opentraj_root, dataset_names)
    for ds_name, ds in datasets.items():
        soc_space_values.append(social_space(ds.data))

    # soc_space_values = histogram_sampler(soc_space_values, max_n_samples=2000, n_bins=100)

    df_social_space = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                               'social_space': soc_space_values[ii],
                                               }) for ii in range(len(dataset_names))])

    print("making social space plots ...")

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(111)
    sns.swarmplot(y='social_space', x='title', data=df_social_space, size=1)

    plt.xlabel('')
    # plt.xticks([])
    plt.xticks(rotation=-90)

    plt.show()
