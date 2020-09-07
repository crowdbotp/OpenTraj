# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import pandas as pd


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
