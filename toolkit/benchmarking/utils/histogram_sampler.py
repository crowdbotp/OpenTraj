# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import sys
import numpy as np
from itertools import chain

"""
This function takes {D}: multiple group of samples (1d) & N(max_n_samples): an integer value
calculate the histogram of each group,
divides the histogram values by a fixed ratio s.t. sum of each histogram < N 
"""


def histogram_sampler(grouped_data: list, quantile_interval=[0.05, 0.95],
                      max_n_samples=1000, n_bins=100):
    # concat all_samples
    all_samples = np.concatenate(grouped_data)
    # drop nan values
    all_samples = all_samples[~np.isnan(all_samples)]
    # take interval of 95% of data (remove outliers)
    data_range = np.quantile(all_samples, quantile_interval)
    # bins for calculating histogram of each group
    hist_bin = np.linspace(data_range[0], data_range[1], n_bins + 1)

    # what is the size of biggest group?
    max_group_size = max([len(g) for g in grouped_data])
    ratio = max_group_size / max_n_samples

    # the input_data after being down-sampled
    grouped_data_sampled = []
    for ii, group in enumerate(grouped_data):
        # calc hist for each group
        group_hist, _ = np.histogram(group, bins=hist_bin)
        # divide each bin by ratio
        group_hist = np.round(group_hist / ratio).astype(int)

        # generate h[i] samples from each bin[i]
        new_samples = [[hist_bin[i]] * group_hist[i] for i in range(n_bins)]
        new_samples = list(chain(*new_samples))
        grouped_data_sampled.append(new_samples)

    return grouped_data_sampled


def normalize_samples_with_histogram(grouped_data: list, quantile_interval=[0.05, 0.95],
                                     max_n_samples=1000, n_bins=100):
    # concat all_samples
    all_samples = np.concatenate(grouped_data)
    # drop nan values
    all_samples = all_samples[~np.isnan(all_samples)]
    # take interval of 95% of data (remove outliers)
    data_range = np.quantile(all_samples, quantile_interval)
    # bins for calculating histogram of each group
    hist_bin = np.linspace(data_range[0], data_range[1], n_bins + 1)

    # the input_data after being down-sampled
    grouped_data_sampled = []
    for ii, group in enumerate(grouped_data):
        # # take interval of 95% of data (remove outliers)
        # data_range = np.quantile(group, quantile_interval)
        # # bins for calculating histogram of each group
        # hist_bin = np.linspace(data_range[0], data_range[1], n_bins + 1)

        # calc hist for each group
        group_hist, _ = np.histogram(group, bins=hist_bin)
        # divide each bin by ratio
        ratio = len(group) / max_n_samples
        group_hist = np.round(group_hist / ratio).astype(int)

        # generate h[i] samples from each bin[i]
        new_samples = [[hist_bin[i]] * group_hist[i] for i in range(n_bins)]
        new_samples = list(chain(*new_samples))
        grouped_data_sampled.append(new_samples)

    return grouped_data_sampled
