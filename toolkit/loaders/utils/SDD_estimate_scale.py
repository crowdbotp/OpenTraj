# Author: Javad Amirian
# Email: amiryan.j@gmail.com

"""
This file tries to estimate the scale of projection of SDD dataset.
By assuming that the trajectories in TrajNet are scaled correctly.
And by matching the trajectories in SDD and Trajnet for each scene.
"""

import os
import sys
import numpy as np

from toolkit.loaders.loader_sdd import load_sdd
from toolkit.loaders.loader_trajnet import load_trajnet

# input the opentraj path here
opentraj_root = sys.argv[1]

trajnet_train_files = [
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/bookstore_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/bookstore_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/bookstore_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/bookstore_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/coupa_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/deathCircle_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/deathCircle_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/deathCircle_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/deathCircle_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/deathCircle_4.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_4.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_5.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_6.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_7.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/gates_8.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/hyang_4.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/hyang_5.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/hyang_6.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/hyang_7.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/hyang_9.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_4.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_7.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_8.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Train/stanford/nexus_9.txt")]

trajnet_test_files = [
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/coupa_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/coupa_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/gates_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/hyang_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/hyang_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/hyang_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/hyang_8.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/little_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/little_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/little_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/little_3.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/nexus_5.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/nexus_6.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/quad_0.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/quad_1.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/quad_2.txt"),
    os.path.join(opentraj_root, "datasets/TrajNet/Test/stanford/quad_3.txt")]

trajnet_files = trajnet_test_files.copy()
trajnet_files.extend(trajnet_train_files)

for trajnet_file in trajnet_files:
    # check if it's a train or test file
    test_file = True
    if 'Train' in trajnet_file:
        test_file = False

    # find the corresponding file in SDD
    scene_name = trajnet_file[trajnet_file.rfind('/') + 1:-6]
    scene_id = int(trajnet_file[-5])
    sdd_file = os.path.join(opentraj_root, 'datasets/SDD/', scene_name, 'video%d' % scene_id, 'annotations.txt')
    if not os.path.exists(sdd_file):
        # print('Error: sdd file does not exist:', sdd_file)
        continue

    # read from trajnet
    trajnet_dataset = load_trajnet(trajnet_file)
    # read from SDD
    sdd_dataset = load_sdd(sdd_file)

    # plot them for manula debug
    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # trajnet_dataset.data.plot.scatter("pos_x", "pos_y", ax=axes[0])
    # sdd_dataset.data.plot.scatter("pos_x", "pos_y", ax=axes[1])
    # plt.show()

    # take one traj from trajnet
    trajnet_ids = trajnet_dataset.get_agent_ids()
    # plt.figure()
    suggested_scale = -1
    for trajnet_id in trajnet_ids:
        # trajnet_id = trajnet_ids[3]
        trajnet_traj = trajnet_dataset.get_trajectories([trajnet_id])[0]
        # trajnet_traj_0.plot.scatter("pos_x", "pos_y", ax=axes[0], color='red')

        trajnet_traj = trajnet_traj[["pos_x", "pos_y"]].dropna().to_numpy()
        # we need a relative trajectory. No?
        trajnet_traj = trajnet_traj - trajnet_traj[0]

        trajnet_traj_size = len(trajnet_traj)
        trajnet_traj_len_L2 = np.linalg.norm(trajnet_traj[-1] - trajnet_traj[0])

        # plt.plot(trajnet_traj[:, 0], trajnet_traj[:, 1], linewidth=3)
        sdd_ids = sdd_dataset.get_agent_ids()
        for sdd_id in sdd_ids:
            sdd_traj_i = sdd_dataset.get_trajectories([sdd_id])[0]
            sdd_traj_i = sdd_traj_i[["pos_x", "pos_y"]].to_numpy()[::12]
            if len(sdd_traj_i) < trajnet_traj_size: continue
            sdd_traj_len = np.linalg.norm(sdd_traj_i[trajnet_traj_size - 1] - sdd_traj_i[0])
            if sdd_traj_len < 1E-8: continue
            suggested_scale = (trajnet_traj_len_L2 / sdd_traj_len)
            sdd_traj_scaled = (sdd_traj_i[:trajnet_traj_size] - sdd_traj_i[0]) * suggested_scale
            error = abs((sdd_traj_scaled - trajnet_traj).sum().sum())

            if 0 < error < 0.02 and 0.020 < suggested_scale < 0.060:
                print(f"{scene_name}:\n\t video{scene_id}:\n\t\tscale: {suggested_scale}\n\t\tcertainty: {1 - error}")
                break

        if 0 < error < 0.02 and 0.020 < suggested_scale < 0.06:
            tf = np.zeros((3, 3), dtype=np.float)
            tf[0, 0], tf[1, 1] = suggested_scale, suggested_scale
            tf[2, 2] = 1
            sdd_dataset.apply_transformation(tf, inplace=True)
            break
        # plt.show()

    # search and find 'scale'
