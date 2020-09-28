import os
import sys
import toolkit.benchmarking.indicators.general_stats as general_stats
import toolkit.benchmarking.indicators.motion_properties as motion_properties
import toolkit.benchmarking.indicators.path_efficiency as path_efficiency
import toolkit.benchmarking.indicators.traj_deviation as traj_deviation
import toolkit.benchmarking.indicators.crowd_density as crowd_density
import toolkit.benchmarking.indicators.collision_energy as collision_energy
import toolkit.benchmarking.indicators.conditional_entropy as conditional_entropy
import toolkit.benchmarking.indicators.trajectory_entropy as trajectory_entropy
import toolkit.benchmarking.indicators.global_multimodality as global_multimodality


if __name__ == "__main__":
    # opentraj_root = sys.argv[1]
    # output_dir = sys.argv[2]

    general_stats.main()
    motion_properties.main()
    path_efficiency.main()
    traj_deviation.main()
    crowd_density.main()
    collision_energy.main()

    # Todo
    conditional_entropy.main()
    trajectory_entropy.main()
    global_multimodality.main()


