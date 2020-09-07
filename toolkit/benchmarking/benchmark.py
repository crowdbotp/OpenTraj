# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import sys
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

from toolkit.loaders.loader_gc import loadGC
from toolkit.benchmarking.metrics.individual.motion import speed #, acceleration
from toolkit.benchmarking.metrics.individual.path_length import path_efficiency
from toolkit.benchmarking.metrics.agent_to_agent.pcf import pcf
from toolkit.benchmarking.metrics.agent_to_agent.distance import social_space

from toolkit.core.trajdataset import TrajDataset
from toolkit.benchmarking.load_all_datasets import get_datasets


def speed_plot(dataset: TrajDataset):
    trajectories = dataset.get_trajectories()
    speeds = []
    for traj in trajectories:
        speeds_i = speed(traj)
        speeds.extend(speeds_i)

    bins = np.linspace(0, 2.5, 25)
    hist, bins, patches = plt.hist(speeds, bins, color='blue', density=True, alpha=0.7)
    # hist, bin_edges = np.histogram(ped_speeds, bins, density=True)

    plt.suptitle(dataset.title)
    plt.ylabel('histogram of speeds')
    plt.xlabel('m/s')
    plt.xlim([bins[0], bins[-1]])
    plt.ylim([0, 6])

    return hist


def acceleration_plot(dataset: TrajDataset):
    trajectories = dataset.get_trajectories()
    accelerations = []
    for traj in trajectories:
        if len(traj) < 2: continue
        accelerations_i = acceleration(traj)
        accelerations.extend(accelerations_i)

    bins = np.linspace(-2.5, 2.5, 100)
    # hist, bin_edges = np.histogram(ped_accelerations, bins, density=True)

    hist, bins, patches = plt.hist(accelerations, bins, color='red', density=True, alpha=0.7)

    plt.title(dataset.title)
    plt.ylabel('histogram of accelerations')
    plt.xlabel('m/s^2')
    plt.xlim([-2.5, 2.5])
    plt.ylim([0, 5])

    return hist


def path_efficiency_plot(dataset: TrajDataset):
    ped_trajectories = dataset.get_trajectories()
    efficiencies = []
    for traj in ped_trajectories:
        if len(traj) < 2: continue

        try:
            p_eff = path_efficiency(traj)
            efficiencies.append(p_eff * 100)
        except:
            print('Error in path efficiency metric')

    bins = np.linspace(50, 100, 25)
    hist, bins, patches = plt.hist(efficiencies, bins, color='pink', density=True, alpha=0.7)
    # hist, bin_edges = np.histogram(efficiencies, bins, density=True)

    plt.title(dataset.title)
    plt.ylabel('path efficiency')
    plt.xlabel('percent')
    plt.xlim([bins[0], bins[-1]])
    plt.ylim([0, 0.5])

    return hist


def density_vanilla_plot(dataset: TrajDataset):  # over observed space of the dataset ([minx,maxx], [miny, maxy])
    space = (dataset.bbox['x']['max'] - dataset.bbox['x']['min']) * \
            (dataset.bbox['y']['max'] - dataset.bbox['y']['min'])
    density_t = []

    frames = dataset.get_frames()
    for frame in frames:
        density_t.append(len(frame) / space)

    bins = np.linspace(0, 1, 40)
    hist, bins, patches = plt.hist(density_t, bins=bins, color='red', density=True, alpha=0.7)

    plt.title(dataset.title)
    plt.ylabel('distribution of density')
    plt.xlabel('Person per m^2')


def density_voronoi_plot(dataset: TrajDataset):  # needs boundaries
    # TODO: find voronoi toolbox python
    return []


def min_dist_plot(dataset: TrajDataset):
    frames = dataset.get_frames()
    min_dists = np.ones(len(frames)) * 1000  # a big number
    for ii, frame in enumerate(frames):
        N_t = len(frame)
        if N_t < 2: continue
        X_t = frame[["pos_x", "pos_y"]].to_numpy()
        # compute distance matrix between all pairs of agents
        DD_t = euclidean_distances(X_t)
        DD_t = DD_t[~np.eye(N_t, dtype=bool)].reshape(N_t, N_t - 1)
        min_DD_t = np.amin(DD_t)
        min_dists[ii] = min_DD_t

    bins = np.linspace(0, 4, 40)
    hist, bins, patches = plt.hist(min_dists, bins, color='green', density=True, alpha=0.7)

    plt.title(dataset.title)
    plt.ylabel('histogram of min distances')
    plt.xlabel('meter')
    plt.xlim([bins[0], bins[-1]])
    plt.ylim([0, 2])

    return hist


def agent_social_space(dataset: TrajDataset):
    sspace = social_space(dataset).values()

    bins = np.linspace(0, 4, 40)
    hist, bins, patches = plt.hist(sspace, bins, color='green', alpha=0.7,
                                   density=True, cumulative=True)

    plt.title(dataset.title)
    plt.ylabel('histogram of social spaces')
    plt.xlabel('meter')
    plt.xlim([bins[0], bins[-1]])
    plt.ylim([0, 2])

    return hist


def pcf_plot(dataset: TrajDataset):
    frames = dataset.get_frames()
    # calc pcf on a dataset
    pcf_accum = []
    pcf_range = np.arange(0.2,  # starting radius
                          8,  # end radius
                          0.2)  # step size

    for frame in frames:
        pcf_values_t = pcf(frame[['pos_x', 'pos_y', 'pos_z']],
                           list(pcf_range), sigma=0.25)
        if not len(pcf_accum):
            pcf_accum = pcf_values_t
        else:
            pcf_accum += pcf_values_t
    avg_pcf = pcf_accum / len(frames)

    plt.title(dataset.title)
    plt.ylabel('PCF')
    plt.xlabel('meter')
    plt.plot(avg_pcf, color='purple')


# Fixme: the program needs the two following inputs:
if __name__ == '__main__':
    opentraj_root = sys.argv[1]  # e.g. os.path.expanduser("~") + '/workspace2/OpenTraj'
    output_dir = sys.argv[2]  # e.g. os.path.expanduser("~") + '/Dropbox/OpenTraj-paper/exp/ver-0.2'
    datasets = get_datasets(opentraj_root)

    # ................................................................

    # ================================================================
    # =================== List the metric functions ==================
    # ================================================================
    metrics = [
        speed_plot,
        # acceleration_plot,
        # path_efficiency_plot,
        # density_vanilla_plot,
        # min_dist_plot,
        # pcf_plot
    ]
    # ================================================================

    for i, dataset in enumerate(datasets):    # ^
        # dataset_entries = dataset.get_entries()     # concatenation of all rows in the dataset
        # dataset_trajs = dataset.get_trajectories()
        # dataset_frames = dataset.get_frames()
        # splitted_trajs = split_trajectories(dataset_trajs)
        # if not splitted_trajs: continue
        # traj_ts = splitted_trajs[0]["timestamp"].to_numpy()
        # pred_dts = traj_ts[:] - traj_ts[0]
        #
        # all_residuals = []
        # for traj in splitted_trajs:
        #     traj_np = traj[["pos_x", "pos_y", "vel_x", "vel_y"]].to_numpy()
        #     constvel_preds = const_vel(traj_np[0], pred_dts)
        #     residuals = constvel_preds - traj_np[:, :2]
        #     all_residuals.append(residuals)
        #
        # all_residuals = np.stack(all_residuals)
        # all_residuals_rs = all_residuals.reshape((-1, 2))
        # cell_size = 0.10  # meter
        # x_edges = np.arange(min(all_residuals_rs[:, 0]), max(all_residuals_rs[:, 0]), cell_size)
        # y_edges = np.arange(min(all_residuals_rs[:, 1]), max(all_residuals_rs[:, 1]), cell_size)
        # res_hist, _, _ = np.histogram2d(all_residuals[:, 1, 0],
        #                                 all_residuals[:, 1, 1],
        #                                 bins=[x_edges, y_edges], density=True)
        #
        # plt.imshow(res_hist, interpolation='nearest', origin='low',
        #            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        # plt.show()

        for j, metric in enumerate(metrics):  # v
            print('metric = ', metric.__name__)
            print(dataset.title)
            plt.close()
            plt.figure()
            # plt.show()
            metric(dataset)

            fig_file = os.path.join(output_dir, metric.__name__, dataset.title + '.png')
            if not os.path.exists(os.path.dirname(fig_file)):
                os.mkdir(os.path.dirname(fig_file))
            plt.savefig(fig_file)
            time.sleep(1)
            plt.clf()
