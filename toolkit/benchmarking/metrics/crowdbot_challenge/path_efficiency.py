from crowdscan.crowd.trajdataset import TrajDataset
from crowdscan.metrics.individual import motion

import numpy
import pandas

# The path efficiency regroups the metrics which evaluate the robot efficiency to reach the goal when it is alone in its environment, 
# compared to its efficiency when it is doing the same task surrounded by a crowd.


# Relative time to goal
# The relative time to goal compares the time taken by the robot to reach its goal when it is alone to the time it takes when it is in a crowd. 
# The metrics is given below: Trtg = Trobot_alone/Tcrowded

# def relative_time_to_goal(reference_trajectory: trajectory, crowded_trajectory: trajectory, main_direction = "pos_x", start_position = -40, end_position = 40):
def relative_time_to_goal(reference_trajectory: TrajectoryDataset, crowded_trajectory: TrajDataset, robot_id: int):
    # actual_traj = reference_trajectory[numpy.logical_and(reference_trajectory[main_direction]>start_position...
    #     ..., reference_trajectory[main_direction]<end_position)].reset_index()['frame id']
    
    # actual_traj = reference_trajectory[reference_trajectory['agent id'] == robot_id].reset_index()['frame id']

    Trobot_alone = reference_trajectory[len(crowded_trajectory)-1] - reference_trajectory[0]
    
    # actual_traj_crowded = crowded_trajectory[numpy.logical_and(crowded_trajectory[main_direction]>start_position...
    #     ..., crowded_trajectory[main_direction]<end_position)].reset_index()['frame id']

    # actual_traj_crowded = crowded_trajectory[reference_trajectory['agent id'] == robot_id].reset_index()['frame id']

    Tcrowded = crowded_trajectory[len(crowded_trajectory)-1] - crowded_trajectory[0]
    
    Trtg = numpy.divide(Trobot_alone,Tcrowded)

    return Trtg

    # Relative path length
    # The relative path length compares the length of the path taken by the robot to reach its goal when a crowd is present or not. 
    # Mathematically it is given by the following formula: Lrp = Lrobot_alone/Lcrowded
def relative_path_length(reference_trajectory: TrajectoryDataset, crowded_trajectory: TrajectoryDataset, robot_id: int):
    reference_trajectory_path_length = motion.path_length(reference_trajectory, robot_id)

    crowded_trajectory_path_length = motion.path_length(crowded_trajectory,robot_id)

    Lrp = numpy.divide(reference_trajectory_path_length, crowded_trajectory_path_length)

    return Lrp

# Relative jerk
# The relative jerk evaluates the smoothness of the path taken by the robot to reach its goal when it is alone compared to when it is in a crowd.
# It is given below:

def relative_jerk(reference_trajectory: TrajectoryDataset, crowded_trajectory: TrajectoryDataset, robot_id: int):
    pass