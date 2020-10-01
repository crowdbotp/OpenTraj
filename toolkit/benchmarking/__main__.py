import os
import sys

from benchmarking.load_all_datasets import get_datasets, get_trajlets, all_dataset_names

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(dir_path, '../..')))

import toolkit.benchmarking.indicators.general_stats as general_stats
import toolkit.benchmarking.indicators.motion_properties as motion_properties
import toolkit.benchmarking.indicators.path_efficiency as path_efficiency
import toolkit.benchmarking.indicators.traj_deviation as traj_deviation
import toolkit.benchmarking.indicators.crowd_density as crowd_density
import toolkit.benchmarking.indicators.collision_energy as collision_energy
import toolkit.benchmarking.indicators.trajectory_entropy as trajectory_entropy
import toolkit.benchmarking.indicators.global_multimodality as global_multimodality


if __name__ == "__main__":
    opentraj_root = sys.argv[1]
    output_dir = sys.argv[2]

    all_datasets = get_datasets(opentraj_root, all_dataset_names)  # map from dataset_name: str -> `TrajDataset` object
    all_trajlets = get_trajlets(opentraj_root, all_dataset_names)  # map from dataset_name: str -> Trajlets (np array)

    general_stats.run(all_datasets, output_dir)
    motion_properties.run(all_trajlets, output_dir)
    path_efficiency.run(all_trajlets, output_dir)
    traj_deviation.run(all_trajlets, output_dir)
    crowd_density.run(all_datasets, output_dir)
    collision_energy.run(all_datasets, output_dir)
    global_multimodality.run(all_trajlets, output_dir)
    trajectory_entropy.run(all_datasets, output_dir)


