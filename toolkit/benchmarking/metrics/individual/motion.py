import numpy as np
import pandas as pd
# from toolkit.core.trajectorydataset import TrajectoryDataset
# from toolkit.core.trajdataset import TrajDataset


def speed(trajectory: pd.DataFrame):
    """
    calc speed (absolute velocity) at each frame! (scalar number)
    :return: an array of 'speed' values
    """
    # TODO: calc velocity if does not exist ?
    speeds = np.linalg.norm(trajectory[['vel_x', 'vel_y']].dropna(), axis=1)
    return speeds


# def acceleration(trajectory: pd.DataFrame):
#     """
#     calc acceleration (absolute value) at each frame! (scalar number)
#     :return: an array of 'acceleration' values
#     """
#     # interval between frames
#     dt = trajectory['timestamp'].diff().dropna().to_numpy()
#     accs = np.diff(speed(trajectory)) / dt
#
#     # padding
#     accs = np.insert(accs, 0, accs[0])
#     # trajectories['acc x'] = trajectories.diff().diff()['pos x']
#     # trajectories['acc y'] = trajectories.diff().diff()['pos y']
#     return accs
#



