# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from bisect import bisect_left, bisect_right


def pcf(locations, bins: list, sigma=0.25, normalize=False):
    """
    Pair Correlation Function: distribution of distances between any pair of agents
    https://homepage.univie.ac.at/Franz.Vesely/simsp/dx/node22.html

    :param locations: location of particles (agents)
    :param bins: sequence of radii where pcf should be computed
    :param sigma: parameter of gaussian kernel
    :param normalize: bool
    :return: 
    """
    def __area_r__(radius):
        return np.pi * (radius + 0.5) ** 2 - np.pi * max((radius - 0.5), 0) ** 2

    def __pcf_r__(dist_matrix, radius, sigma, normalize):
        dists_sqr = np.power(dist_matrix - radius, 2)
        dists_exp = np.exp(-dists_sqr / (sigma ** 2)) / (np.sqrt(np.pi) * sigma)
        pcf_r = np.sum(dists_exp) / __area_r__(radius)
        if normalize:
            pcf_r /= (2.0 * dist_matrix.shape[1])
            pcf_r /= dist_matrix.shape[0]  # normalize B
        return pcf_r

    N = len(locations)
    pcf_values = np.zeros(len(bins), dtype=np.float)
    # self.get_pcf = lambda r: pcf_values[bisect_right(rad_values, r)]
    if N < 2: return pcf_values

    dists = euclidean_distances(locations)
    dists_wo_diag = dists[~np.eye(N, dtype=bool)].reshape(N, N - 1)

    for ii, rr in enumerate(bins):
        pcf_values[ii] = __pcf_r__(dists_wo_diag, rr, sigma, normalize=normalize)
    return pcf_values
