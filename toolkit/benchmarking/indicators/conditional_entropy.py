# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from toolkit.core.trajdataset import TrajDataset
import numpy as np
from numpy.linalg.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances


def gaussian_kernel(x1, x2):
    return np.exp(-norm(x1 - x2) ** 2 / 2)


def conditional_entropy(trajlets: list):
    # H(Xn) = E_xp(ln(p( Xp | Xn )))
    # p(Xp|Xn)  = p(Xp, Xn) / p(Xn) = p(X) / p(Xn)

    euclidean_distances(trajlets)

    cE = 0
    for traj_i in trajlets:
        xi = traj_i.whole
        xoi = traj_i.obsv
        xpi = traj_i.pred
        for traj_j in trajlets:
            xj = traj_j.whole
            xoj = traj_j.obsv
            xpj = traj_j.pred
            # K_xi_xj = np.exp(-norm(x1 - x2) ** 2 / 2)

    return cE




if __name__ == "__main__":
    from toolkit.loaders.loader_eth import load_eth
    from toolkit.core.trajlet import split_trajectories, to_numpy

    eth_dataset = load_eth("/home/cyrus/workspace2/OpenTraj/datasets/ETH/seq_eth/obsmat.txt")
    eth_trajs = eth_dataset.get_trajectories("pedestrian")
    eth_trajlets = split_trajectories(eth_trajs)
    to_numpy(eth_trajlets)

    eth_trajlets = np.stack(eth_trajlets)
    conditional_entropy(eth_trajlets)
