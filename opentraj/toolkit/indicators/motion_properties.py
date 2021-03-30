# Author: Javad Amirian
# Email: amiryan.j@gmail.com
import os
import sys
import numpy as np
import pandas as pd
from toolkit.utils.histogram_sampler import normalize_samples_with_histogram
import matplotlib.pyplot as plt
import seaborn as sns


def speed_var_of_trajs(trajlets_np: np.ndarray):
    """
    :param trajlets_np: numpy array in form of [N, T, 4] => N is number of trajlets, and T is the number of frames
    """
    speed_values = np.linalg.norm(trajlets_np[:, :, 2:4], axis=2)
    speed_var_samples = np.max(speed_values[::1], axis=1) - np.min(speed_values[::1], axis=1)
    return speed_var_samples


def speed_avg_of_trajs(trajlets_np: np.ndarray):
    speed_values = np.linalg.norm(trajlets_np[:, :, 2:4], axis=2)
    speed_avg_samples = np.mean(speed_values[::1], axis=1)
    return speed_avg_samples


def acceleration_of_tarjs(trajlets_np: np.ndarray):
    speed_values = np.linalg.norm(trajlets_np[:, :, 2:4], axis=2)
    dt = np.diff(trajlets_np[:, :, 4], axis=1)
    acc_values = np.diff(speed_values, axis=1) / dt
    # acc_values = np.abs(acc_values)  # FixMe
    return np.mean(np.abs(acc_values), axis=1), np.max(np.abs(acc_values), axis=1)


def run(trajlets, output_dir):
    print("calculating speed_var indicator ...")
    speed_mean_values, speed_var_values = [], []
    dataset_names = list(trajlets.keys())
    for ds_name in dataset_names:
        speed_mean_values.append(speed_avg_of_trajs(trajlets[ds_name]))
        speed_var_values.append(speed_var_of_trajs(trajlets[ds_name]))

    print("calculating acceleration indicator ...")
    acc_mean_values, acc_max_values = [], []
    for ds_name in dataset_names:
        acc_mean_values_i, acc_max_values_i = acceleration_of_tarjs(trajlets[ds_name])
        acc_mean_values.append(acc_mean_values_i)
        acc_max_values.append(acc_max_values_i)

    # down-sample values
    # down-sample each group. Why?
    # Because the samples are too much and swarm-plot can not handle this many samples!
    speed_mean_values = normalize_samples_with_histogram(speed_mean_values, max_n_samples=300, n_bins=50, quantile_interval=[0, 0.92])
    speed_var_values = normalize_samples_with_histogram(speed_var_values, max_n_samples=300, n_bins=50, quantile_interval=[0, 0.92])
    acc_mean_values = normalize_samples_with_histogram(acc_mean_values, max_n_samples=300, n_bins=50, quantile_interval=[0.02, 0.92])
    acc_max_values = normalize_samples_with_histogram(acc_max_values, max_n_samples=300, n_bins=50, quantile_interval=[0.02, 0.92])

    # put samples in a DataFrame (required for seaborn plots)
    df_speed_mean = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                            'speed_mean': speed_mean_values[ii],
                                             }) for ii in range(len(trajlets))])

    df_speed_var = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                            'speed_var': speed_var_values[ii],
                                            }) for ii in range(len(trajlets))])

    df_acc_mean = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                           'acc_mean': acc_mean_values[ii],
                                           }) for ii in range(len(trajlets))])

    df_acc_max = pd.concat([pd.DataFrame({'title': dataset_names[ii],
                                          'acc_max': acc_max_values[ii],
                                          }) for ii in range(len(trajlets))])

    print("producing motion plots ...")

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(411)
    sns.swarmplot(y='speed_mean', x='title', data=df_speed_mean, size=1)
    plt.ylim([0, 2.4])
    plt.xlabel('')
    plt.xticks([])
    ax1.set_yticks([0, 0.5, 1, 1.5, 2.])
    ax1.yaxis.label.set_size(9)
    ax1.yaxis.set_tick_params(labelsize=8)

    ax2 = fig.add_subplot(412)
    sns.swarmplot(y='speed_var', x='title', data=df_speed_var, size=1)
    plt.ylim([0, 2.])
    plt.xlabel('')
    plt.xticks([])
    ax2.set_yticks([0, 0.5, 1, 1.5, 2.])
    ax2.yaxis.label.set_size(9)
    ax2.yaxis.set_tick_params(labelsize=8)

    ax3 = fig.add_subplot(413)
    sns.swarmplot(y='acc_mean', x='title', data=df_acc_mean, size=1)
    plt.ylim([0, 0.5])
    plt.xlabel('')
    plt.xticks([])
    ax3.yaxis.label.set_size(9)
    ax3.yaxis.set_tick_params(labelsize=8)

    ax4 = fig.add_subplot(414)
    sns.swarmplot(y='acc_max', x='title', data=df_acc_max, size=1)
    plt.ylim([0, 1.2])
    plt.xlabel('')
    plt.xticks(rotation=-20)
    ax4.yaxis.label.set_size(9)
    ax4.yaxis.set_tick_params(labelsize=8)
    ax4.xaxis.set_tick_params(labelsize=8)

    plt.savefig(os.path.join(output_dir, 'motion_properties.pdf'), dpi=500, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    from toolkit.test.load_all import all_dataset_names, get_datasets, get_trajlets

    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]

    all_datasets = get_datasets(opentraj_root, all_dataset_names)
    all_trajlets = get_trajlets(opentraj_root, all_dataset_names)

    run(all_trajlets, output_dir)
