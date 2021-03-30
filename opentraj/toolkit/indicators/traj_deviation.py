# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
# matplotlib.use('PS')
from toolkit.core.trajdataset import TrajDataset


def deviation_from_linear_pred(trajlets, save_plots_to):
    dp_from_t0 = trajlets[:, :, :2] - np.expand_dims(trajlets[:, 0, :2], 1)
    first_significant_displacement_idx = (np.linalg.norm(dp_from_t0, axis=2) > 0.25).argmax(axis=1)
    first_significant_displacement = np.stack([dp_from_t0[i, first_significant_displacement_idx[i], :2]
                                               for i in range(len(trajlets))])
    # start_thetas = np.arctan2(trajlets[:, 0, 2], trajlets[:, 0, 3])  # calculated from first velocity vector
    # start_thetas = np.arctan2(trajlets[:, 2, 0] - trajlets[:, 0, 0],
    #                           trajlets[:, 2, 1] - trajlets[:, 0, 1])
    start_thetas = np.arctan2(first_significant_displacement[:, 0], first_significant_displacement[:, 1])

    rot_matrices = np.stack([np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]]) for theta in start_thetas])
    trajs_zero_based = trajlets[:, :, :2] - trajlets[:, 0, :2].reshape((-1, 1, 2))

    trajs_aligned = np.matmul(rot_matrices, trajs_zero_based.transpose((0, 2, 1))).transpose((0, 2, 1))
    is_nan = ~np.any(np.any(np.isnan(trajs_aligned), axis=2), axis=1)
    trajs_aligned = trajs_aligned[is_nan, :, :]

    keypoints = np.mean(trajs_aligned[:, :, :], axis=0)
    keypoints_radius = np.linalg.norm(keypoints, axis=1)
    keypoints_dev_avg = np.rad2deg(np.arctan2(keypoints[:, 0], keypoints[:, 1]))
    keypoints_dev_std = np.std(np.rad2deg(np.arctan2(trajs_aligned[:, :, 0],
                                                     trajs_aligned[:, :, 1])), axis=0)

    # ======== PLOT ============
    fig1, ax1 = plt.subplots()
    trajs_plt = ax1.plot(trajs_aligned[:, :, 1].T, trajs_aligned[:, :, 0].T, alpha=0.3, color='blue')
    avg_plt = ax1.plot(keypoints[::2, 1], keypoints[::2, 0], 'o', color='red')

    for ii in range(2, len(keypoints), 2):
        arc_i = patches.Arc([0, 0], zorder=10,
                            width=keypoints_radius[ii] * 2,
                            height=keypoints_radius[ii] * 2,
                            theta1=keypoints_dev_avg[ii] - keypoints_dev_std[ii],
                            theta2=keypoints_dev_avg[ii] + keypoints_dev_std[ii])
        ax1.add_patch(arc_i)

    ax1.grid()
    ax1.set_aspect('equal')
    plt.title(ds_name)
    plt.xlim([-1.5, 10])
    plt.ylim([-4, 4])
    plt.legend(handles=[trajs_plt[0], avg_plt[0]],
               labels=["trajlets", "avg"], loc="lower left")

    plt.savefig(os.path.join(save_plots_to, 'dev-' + ds_name + '.png'))
    # plt.show()

    return keypoints_dev_avg, keypoints_dev_std


def run(trajlets, output_dir):
    global ds_name
    # dataset_names = list(trajlets.keys())

    deviation_stats = {1.6: [], 2.4: [], 4.8: []}
    for ds_name in dataset_names:
        dev_avg, dev_std = deviation_from_linear_pred(trajlets[ds_name], output_dir)

        for t in [1.6, 2.4, 4.8]:
            dt = np.diff(trajlets[ds_name][0, :, 4])[0]
            time_index = int(round(t/dt))-1
            deviation_stats[t].append([dev_avg[time_index], dev_std[time_index]])

    deviation_stats[1.6] = np.array(deviation_stats[1.6])
    deviation_stats[2.4] = np.array(deviation_stats[2.4])
    deviation_stats[4.8] = np.array(deviation_stats[4.8])

    fig = plt.figure(figsize=(len(dataset_names)+2, 4))

    ax1 = fig.add_subplot(211)
    plt.bar(np.arange(len(dataset_names)), deviation_stats[4.8][:, 0],
            yerr=deviation_stats[4.8][:, 1], alpha=0.7, color='red',
            error_kw=dict(ecolor='blue', lw=2, capsize=5, capthick=2))
    plt.xticks([])
    plt.yticks([-30, -15, 0, 15, 30], ['$-30^o$', '$-15^o$', '$0^o$', '$15^o$', '$30^o$'])
    # plt.yticks([-30, 0, 30])
    plt.grid(axis='y', linestyle='--')
    plt.ylabel('$t=4.8s$')

    ax2 = fig.add_subplot(212)
    plt.bar(np.arange(len(dataset_names)), abs(deviation_stats[2.4][:, 0]),
            yerr=deviation_stats[2.4][:, 1], alpha=0.7, color='red',
            error_kw=dict(ecolor='blue', lw=2, capsize=5, capthick=2))
    plt.xticks([])
    plt.yticks([-20, -10, 0, 10, 20], ['$-20^o$', '$-10^o$', '$0^o$', '$10^o$', '$20^o$'])
    plt.grid(axis='y', linestyle='--')
    plt.ylabel('$t=2.4s$')

    # ax3 = fig.add_subplot(313)
    # plt.bar(np.arange(len(dataset_names)), abs(deviation_stats[1.6][:, 0]),
    #        yerr=deviation_stats[1.6][:, 1], alpha=0.5,
    #        error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    # plt.ylabel('t=1.6s')

    plt.xticks(np.arange(0, len(dataset_names), 1.0))
    ax2.set_xticklabels(dataset_names)
    ax2.xaxis.set_tick_params(labelsize=8)
    plt.xticks(rotation=-20)
    # ax1.margins(0.05)

    plt.subplots_adjust(wspace=0, hspace=.10)
    plt.savefig(os.path.join(output_dir, 'traj_deviation.pdf'), dpi=400, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    from toolkit.test.load_all import all_dataset_names, get_trajlets

    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]

    # dataset_names = ['KITTI']
    dataset_names = ['ETH-Univ', 'ETH-Hotel', 'UCY-Zara', 'UCY-Univ']
    # dataset_names = all_dataset_names
    trajlets = get_trajlets(opentraj_root, dataset_names)
    run(trajlets, output_dir)

