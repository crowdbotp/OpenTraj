# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def social_space(dataset: pd.DataFrame):
    """The minimum distance that each of agent see during their lifetime"""
    social_spaces = {}  # a map from agent_id => social space
    dataset["min_dist"] = 1000  # default value : a big number

    frame_ids = pd.unique(dataset["frame_id"])
    for frame_id in frame_ids:
        # table indices at t = frame_id
        indices_t = (dataset["frame_id"] == frame_id)

        # get all positions at t = frame_id
        X_t = dataset[["pos_x", "pos_y"]].loc[indices_t]
        N_t = len(X_t)  # number of agents at t = frame_id
        if N_t > 1:
            # agent_ids_t = dataset["agent_id"].loc[indices_t]

            # compute distance matrix between all pairs of agents
            DD_t = euclidean_distances(X_t)

            # remove the diagonal elements (or self-distances)
            DD_t = DD_t[~np.eye(N_t, dtype=bool)].reshape(N_t, N_t - 1)

            # calc min dist for each agent at t = frame_id
            minD_t = np.min(DD_t, axis=1)
            dataset["min_dist"].loc[indices_t] = minD_t

    agent_ids = pd.unique(dataset["agent_id"])
    for agent_id in agent_ids:
        agent_dists = dataset["min_dist"].loc[dataset["agent_id"] == agent_id].to_numpy()
        social_spaces[agent_id] = min(agent_dists)

    return social_spaces
