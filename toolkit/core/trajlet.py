# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import numpy as np


def split_trajectories(traj_groups, length=4.8, overlap=2., static_filter_thresh=1., to_numpy=False):
    """
    :param traj_groups: DataFrameGroupBy containing N group for N agents
    :param length:      min duration for trajlets
    :param overlap:     min overlap duration between consequent trajlets
    :param static_filter_thresh:  if a trajlet is shorter than this thrshold, then it is static
    :param to_numpy: (bool) if True the result will be np.ndarray
    :return: list of Pandas DataFrames (all columns)
             or Numpy ndarray(NxTx5): ["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]
    """

    trajlets = []
    trajs = [g for _, g in traj_groups]
    ts = trajs[0]["timestamp"]
    dt = ts.iloc[1] - ts.iloc[0]
    eps = 1E-2

    for tr in trajs:
        if len(tr) < 2: continue

        f_per_traj = int(np.ceil((length - eps) / dt))
        f_step = int(np.ceil((length - overlap - eps) / dt))

        n_frames = len(tr)
        for start_f in range(0, n_frames - f_per_traj, f_step):
            if static_filter_thresh < 1E-3 or \
                    np.linalg.norm(tr[["pos_x", "pos_y"]].iloc[start_f + f_per_traj].to_numpy() -
                                   tr[["pos_x", "pos_y"]].iloc[start_f].to_numpy()) > static_filter_thresh:
                trajlets.append(tr.iloc[start_f:start_f + f_per_traj])
            # else:
            #     print('removed short trajlet: ',
            #           np.linalg.norm(tr[["pos_x", "pos_y"]].iloc[start_f + f_per_traj].to_numpy() -
            #                          tr[["pos_x", "pos_y"]].iloc[start_f].to_numpy()))

    if to_numpy:
        trl_np_list = []
        for trl in trajlets:
            trl_np = trl[["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]].to_numpy()
            trl_np_list.append(trl_np)
        trajlets = np.stack(trl_np_list)

    return trajlets


def split_trajectories_paired(traj_groups, length=4.8, overlap=2., static_filter_thresh=1., to_numpy=False):
    """
    :param traj_groups: DataFrameGroupBy containing N group for N agents
    :param length:      min duration for trajlets
    :param overlap:     min overlap duration between consequent trajlets
    :param static_filter_thresh:  if a trajlet is shorter than this thrshold, then it is static
    :param to_numpy: (bool) if True the result will be np.ndarray
    :return: list of Pandas DataFrames (all columns)
             or Numpy ndarray(Nx2xTx5): ["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]
    """

    trajlets = []
    trajs = [g for _, g in traj_groups]
    ts = trajs[0]["timestamp"]
    dt = ts.iloc[1] - ts.iloc[0]
    eps = 1E-2

    for tr in trajs:
        if len(tr) < 2: continue

        f_per_traj = int(np.ceil((length - eps) / dt))
        f_step = int(np.ceil((length - overlap - eps) / dt))

        n_frames = len(tr)
        for start_f in range(0, n_frames - f_per_traj*2, f_step):
            if static_filter_thresh < 1E-3 or \
                    np.linalg.norm(tr[["pos_x", "pos_y"]].iloc[start_f + f_per_traj*2].to_numpy() -
                                   tr[["pos_x", "pos_y"]].iloc[start_f].to_numpy()) > static_filter_thresh:
                trajlets.append([tr.iloc[start_f:start_f + f_per_traj],
                                 tr.iloc[start_f+f_per_traj:start_f + 2*f_per_traj]])

    if to_numpy:
        trl_np_list = []
        for trl_obsv, trl_pred in trajlets:
            trl_obsv_np = trl_obsv[["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]].to_numpy()
            trl_pred_np = trl_pred[["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]].to_numpy()
            trl_np_list.append([trl_obsv_np, trl_pred_np])
        trajlets = np.stack(trl_np_list)

    return trajlets


# test
if __name__ == "__main__":
    from toolkit.loaders.loader_eth import load_eth
    import sys, os
    opentraj_root = sys.argv[1]
    # test_dataset = loadETH(os.path.join(opentraj_root, "datasets/ETH/seq_eth/obsmat.txt"))
    test_dataset = load_eth(os.path.join(opentraj_root, "datasets/ETH/seq_hotel/obsmat.txt"))
    trajs = test_dataset.get_trajectories()
    trajlets_4_8s = split_trajectories(trajs, length=4.8, to_numpy=True)
    trajlets_8s = split_trajectories(trajs, length=8, to_numpy=True)
    paired_trajlets_4_8s = split_trajectories_paired(trajs, length=4.8, to_numpy=True)

    print("Test hotel dataset\n******************")
    print("trajlets_4_8s.shape =", trajlets_4_8s.shape)
    print("trajlets_8s.shape =", trajlets_8s.shape)
    print("paired_trajlets_4_8s.shape =", paired_trajlets_4_8s.shape)
