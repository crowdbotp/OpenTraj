# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np
import pandas as pd
from tqdm import tqdm
from opentraj.toolkit.utils.kalman_smoother import KalmanModel
pd.options.mode.chained_assignment = None  # default='warn'


class TrajDataset:
    def __init__(self):
        """
        data might include the following columns:
        "scene_id", "frame_id", "agent_id",
         "pos_x", "pos_y"
          "vel_x", "vel_y",
        """
        self.critical_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
        self.data = pd.DataFrame(columns=self.critical_columns)

        # a map from agent_id to a list of [agent_ids] that are annotated as her groupmate
        # if such informatoin is not available the map should be filled with an empty list
        # for each agent_id
        self.groupmates = {}

        # fps is necessary to calc any data related to time (e.g. velocity, acceleration)

        self.title = ''
        self.fps = -1

        # bounding box of trajectories
        #  FixME: bbox should be a function of scene_id
        # self.bbox = pd.DataFrame({'x': [np.nan, np.nan],
        #                           'y': [np.nan, np.nan]},
        #                          index=['min', 'max'])

        # FixMe ?
        #  self.trajectories_lazy = []

    def postprocess(self, fps, sampling_rate=1, use_kalman=False):
        """
        This function should be called after loading the data by loader
        It performs the following steps:
        -: check fps value, should be set and bigger than 0
        -: check critical columns should exist in the table
        -: update data types
        -: fill 'groumates' if they are not set
        -: checks if velocity do not exist, compute it for each agent
        -: compute bounding box of trajectories

        :param fps: video framerate
        :param sampling_rate: if bigger than one, the data needs downsampling,
                              otherwise needs interpolation
        :param use_kalman:  for smoothing agent velocities
        :return: None
        """

        # check
        for critical_column in self.critical_columns:
            if critical_column not in self.data:
                raise ValueError("Error! some critical columns are missing from trajectory dataset!")

        # modify data types
        self.data["frame_id"] = self.data["frame_id"].astype(int)
        if str(self.data["agent_id"].iloc[0]).replace('.', '', 1).isdigit():
            self.data["agent_id"] = self.data["agent_id"].astype(int)
        self.data["pos_x"] = self.data["pos_x"].astype(float)
        self.data["pos_y"] = self.data["pos_y"].astype(float)
        self.data["label"] = self.data["label"].str.lower()  # search with lower-case labels

        # fill scene_id
        if "scene_id" not in self.data:
            self.data["scene_id"] = 0
        self.fps = fps


        # fill timestamps based on frame_id and video_fps
        if "timestamp" not in self.data:
            self.data["timestamp"] = self.data["frame_id"] / fps

        # fill groupmates
        agent_ids = pd.unique(self.data["agent_id"])
        for agent_id in agent_ids:
            if agent_id not in self.groupmates:
                self.groupmates[agent_id] = []

        # down/up sampling frames
        if sampling_rate >= 2:
            # FixMe: down-sampling
            sampling_rate = int(sampling_rate)
            self.data = self.data.loc[(self.data["frame_id"] % sampling_rate) == 0]
            self.data = self.data.reset_index()
        elif sampling_rate < (1-1E-2):
            # TODO: interpolation
            pass
        else:pass

        # remove the trajectories shorter than 2 frames
        data_grouped = self.data.groupby(["scene_id", "agent_id"])
        single_length_inds = data_grouped.head(1).index[data_grouped.size() < 2]
        self.data = self.data.drop(single_length_inds)

        # fill velocities
        if "vel_x" not in self.data:
            data_grouped = self.data.groupby(["scene_id", "agent_id"])
            dt = data_grouped["timestamp"].diff()

            if (dt > 2).sum():
                print('Warning! too big dt in [%s]' % self.title)

            self.data["vel_x"] = (data_grouped["pos_x"].diff() / dt).astype(float)
            self.data["vel_y"] = (data_grouped["pos_y"].diff() / dt).astype(float)
            nan_inds = np.array(np.nonzero(dt.isnull().to_numpy())).reshape(-1)
            self.data["vel_x"].iloc[nan_inds] = self.data["vel_x"].iloc[nan_inds + 1].to_numpy()
            self.data["vel_y"].iloc[nan_inds] = self.data["vel_y"].iloc[nan_inds + 1].to_numpy()

        # ============================================
        if use_kalman:
            def smooth(group):
                if len(group) < 2: return group
                dt = group["timestamp"].diff().iloc[1]
                kf = KalmanModel(dt, n_dim=2, n_iter=7)
                smoothed_pos, smoothed_vel = kf.smooth(group[["pos_x", "pos_y"]].to_numpy())
                group["pos_x"] = smoothed_pos[:, 0]
                group["pos_y"] = smoothed_pos[:, 1]

                group["vel_x"] = smoothed_vel[:, 0]
                group["vel_y"] = smoothed_vel[:, 1]
                return group

            tqdm.pandas(desc="Smoothing trajectories (%s)" % self.title)
            # print('Smoothing trajectories ...')
            data_grouped = self.data.groupby(["scene_id", "agent_id"])
            self.data = data_grouped.progress_apply(smooth)

        # compute bounding box
        # Warning: the trajectories should belong to the same (physical) scene
        # self.bbox['x']['min'] = min(self.data["pos_x"])
        # self.bbox['x']['max'] = max(self.data["pos_x"])
        # self.bbox['y']['min'] = min(self.data["pos_y"])
        # self.bbox['y']['max'] = max(self.data["pos_y"])

    def interpolate_frames(self, inplace=True):
        """
        Knowing the framerate , the FRAMES that are not annotated will be interpolated.
        :param inplace: Todo
        :return: None
        """
        all_frame_ids = sorted(pd.unique(self.data["frame_id"]))
        if len(all_frame_ids) < 2:
            # FixMe: print warning
            return

        frame_id_A = all_frame_ids[0]
        frame_A = self.data.loc[self.data["frame_id"] == frame_id_A]
        agent_ids_A = frame_A["agent_id"].to_list()
        interp_data = self.data  # "agent_id", "pos_x", "pos_y", "vel_x", "vel_y"
        # df.append([df_try] * 5, ignore_index=True
        for frame_id_B in tqdm(all_frame_ids[1:], desc="Interpolating frames"):
            frame_B = self.data.loc[self.data["frame_id"] == frame_id_B]
            agent_ids_B = frame_B["agent_id"].to_list()

            common_agent_ids = list(set(agent_ids_A) & set(agent_ids_B))
            frame_A_fil = frame_A.loc[frame_A["agent_id"].isin(common_agent_ids)]
            frame_B_fil = frame_B.loc[frame_B["agent_id"].isin(common_agent_ids)]
            for new_frame_id in range(frame_id_A+1, frame_id_B):
                alpha = (new_frame_id - frame_id_A) / (frame_id_B - frame_id_A)
                new_frame = frame_A_fil.copy()
                new_frame["frame_id"] = new_frame_id
                new_frame["pos_x"] = frame_A_fil["pos_x"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["pos_x"].to_numpy() * alpha
                new_frame["pos_y"] = frame_A_fil["pos_y"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["pos_y"].to_numpy() * alpha
                new_frame["vel_x"] = frame_A_fil["vel_x"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["vel_x"].to_numpy() * alpha
                new_frame["vel_y"] = frame_A_fil["vel_y"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["vel_y"].to_numpy() * alpha
                if inplace:
                    self.data = self.data.append(new_frame)
                else:
                    self.data = self.data.append(new_frame)  # TODO
            frame_id_A = frame_id_B
            frame_A = frame_B
            agent_ids_A = agent_ids_B
        self.data = self.data.sort_values('frame_id')

    def apply_transformation(self, tf: np.ndarray, inplace=False):
        """
        :param tf: np.ndarray
            Homogeneous Transformation Matrix,
            3x3 for 2D data
        :param inplace: bool, default False
            If True, do operation inplace
        :return: transformed data table
        """
        if inplace:
            target_data = self.data
        else:
            target_data = self.data.copy()

        # data is 2D
        assert tf.shape == (3, 3)
        tf = tf[:2, :]  # remove the last row
        poss = target_data[["pos_x", "pos_y"]].to_numpy(dtype=np.float)
        poss = np.concatenate([poss, np.ones((len(poss), 1))], axis=1)
        target_data[["pos_x", "pos_y"]] = np.matmul(tf, poss.T).T

        # apply on velocities
        tf[:, -1] = 0  # do not apply the translation element on velocities!
        vels = target_data[["vel_x", "vel_y"]].to_numpy(dtype=np.float)
        vels = np.concatenate([vels, np.ones((len(vels), 1))], axis=1)
        target_data[["vel_x", "vel_y"]] = np.matmul(tf, vels.T).T

        return target_data

    # FixMe: rename to add_row()/add_entry()
    def add_agent(self, agent_id, frame_id, pos_x, pos_y):
        """Add one single data at a specific frame to dataset"""
        new_df = pd.DataFrame(columns=self.critical_columns)
        new_df["frame_id"] = [int(frame_id)]
        new_df["agent_id"] = [int(agent_id)]
        new_df["pos_x"] = [float(pos_x)]
        new_df["pos_y"] = [float(pos_y)]
        self.data = self.data.append(new_df)

    def get_agent_ids(self):
        """:return all agent_id in data table"""
        return pd.unique(self.data["agent_id"])

    # TODO:
    def get_entries(self, agent_ids=[], frame_ids=[], label=""):
        """
        Returns a list of data entries
        :param agent_ids: select specific agent ids, ignore if empty
        :param frame_ids: select a time interval, ignore if empty  # TODO:
        :param label: select agents from a specific label (e.g. car), ignore if empty # TODO:
        :return list of data entries
        """
        output_table = self.data  # no filter
        if agent_ids:
            output_table = output_table[output_table["agent_id"].isin(agent_ids)]
        if frame_ids:
            output_table = output_table[output_table["frame_id"].isin(frame_ids)]
        return output_table

    def get_frames(self, frame_ids: list = [], scene_ids=[]):
        if not len(frame_ids):
            frame_ids = pd.unique(self.data["frame_id"])
        if not len(scene_ids):
            scene_ids = pd.unique(self.data["scene_id"])

        frames = []
        for scene_id in scene_ids:
            for frame_id in frame_ids:
                frame_df = self.data.loc[(self.data["frame_id"] == frame_id) &
                                         (self.data["scene_id"] == scene_id)]
                # traj_df = self.data.filter()
                frames.append(frame_df)
        return frames

    def get_trajectories(self, label=""):
        """
        Returns a list of trajectories
        :param label: select agents from a specific class (e.g. pedestrian), ignore if empty
        :return list of trajectories
        """

        trajectories = []
        df = self.data
        if label:
            label_filtered = self.data.groupby("label")
            df = label_filtered.get_group(label.lower())

        return df.groupby(["scene_id", "agent_id"])

    def get_trajlets(self, length=4.8, overlap=2., filter_min_displacement=1., to_numpy=False):
        """
            :param length:      min duration for trajlets
            :param overlap:     min overlap duration between consequent trajlets
            :param filter_min_displacement:  if a trajlet is shorter than this thrshold, then it is stationary
                                             and would be filtered
            :param to_numpy: (bool) if True the result will be np.ndarray
            :return: list of Pandas DataFrames (all columns)
                     or Numpy ndarray(NxTx5): ["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]
        """

        trajs = self.get_trajectories()
        trajlets = []

        EPS = 1E-2
        dt = trajs["timestamp"].diff().dropna().iloc[0]
        f_per_traj = int(np.ceil((length - EPS) / dt))
        f_step = int(np.ceil((length - overlap - EPS) / dt))

        for _, tr in trajs:
            if len(tr) < 2: continue

            n_frames = len(tr)
            for start_f in range(0, n_frames - f_per_traj, f_step):
                if filter_min_displacement < EPS or \
                        np.linalg.norm(tr[["pos_x", "pos_y"]].iloc[start_f + f_per_traj].to_numpy() -
                                       tr[["pos_x", "pos_y"]].iloc[start_f].to_numpy()) > filter_min_displacement:
                    trajlets.append(tr.iloc[start_f:start_f + f_per_traj])

        if to_numpy:
            trl_np_list = []
            for trl in trajlets:
                trl_np = trl[["pos_x", "pos_y", "vel_x", "vel_y", "timestamp"]].to_numpy()
                trl_np_list.append(trl_np)
            trajlets = np.stack(trl_np_list)

        return trajlets

    # Todo: add 'to_numpy' arguement
    def get_simultaneous_trajlets(self, length=8.0, overlap=4.0):
        ts = self.data["timestamp"].min()
        te = self.data["timestamp"].max()

        EPS = 1E-2
        trajlets = []
        for t_start_i in np.arange(ts, te, length-overlap):
            t_end_i = t_start_i + length
            trimmed_data = self.data.loc[(self.data["timestamp"] >= t_start_i) &
                                         (self.data["timestamp"] < t_end_i - EPS)]
            if len(trimmed_data) == 0:
                continue
            grouped_trajs = trimmed_data.groupby("agent_id")
            dt = grouped_trajs["timestamp"].diff().dropna().iloc[0]
            trajlets.append([gr for _, gr in grouped_trajs])

            # todo
            max_nb_frames = length / dt

        return trajlets


def merge_datasets(dataset_list, new_title=[]):
    if len(dataset_list) < 1:
        return TrajDataset()
    elif len(dataset_list) == 1:
        return dataset_list[0]

    merged = dataset_list[0]

    for ii in range(1, len(dataset_list)):
        merged.data = merged.data.append(dataset_list[ii].data)
        if dataset_list[ii].title != merged.title:
            merged.title = merged.title + " + " + dataset_list[ii].title
    if len(new_title):
        merged.title = new_title

    return merged


