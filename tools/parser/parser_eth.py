import os
import sys
import numpy as np


class ParserETH:
    """
        Parser class for ETH dataset and UCY dataset
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation file: e.g. "OpenTraj/ETH/seq_eth/obsmat.txt"

        Attributes:
            id_p_dict: map from pedestrian id to her trajectory (positions)
            id_v_dict: map from pedestrian id to her velocity data
            id_t_dict: map from pedestrian id to timestamps she appears
            t_id_dict: map from dataset timestamps to pedestrian ids
            t_p_dict : map from dataset timestamps to location of all pedestrians
            min_t    : first timestamp
            max_t    : last timestamp
            interval : interval between timestamps
            [min_x, max_x], [min_y, max_y] : spacial extents of all the trajectories

    """

    def __init__(self, filename=''):
        self.id_p_dict = dict()
        self.id_v_dict = dict()
        self.id_t_dict = dict()
        self.id_label_dict = dict()  # FIXME
        self.t_id_dict = dict()
        self.t_p_dict = dict()
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = -1
        self.min_x = 1000
        self.min_y = 1000
        self.max_x = 0
        self.max_y = 0
        if filename:
            self.load(filename)

    def load(self, filename, down_sample=1, delimit=' '):

        # to search for files in a folder?
        file_names = list()
        if '*' in filename:
            files_path = filename[:filename.index('*')]
            extension = filename[filename.index('*') + 1:]
            for file in os.listdir(files_path):
                if file.endswith(extension):
                    file_names.append(files_path + file)
        else:
            file_names.append(filename)

        for file in file_names:
            if not os.path.exists(file):
                raise ValueError("No such file or directory:", file)
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                for i, row in enumerate(content):
                    row = row.split(delimit)
                    while '' in row: row.remove('')
                    if len(row) < 8: continue

                    ts = int(float(row[0]))
                    id = round(float(row[1]))
                    if ts % down_sample != 0:
                        continue

                    px = float(row[2])
                    py = float(row[4])
                    vx = float(row[5])
                    vy = float(row[7])

                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts
                    if px < self.min_x: self.min_x = px
                    if px > self.max_x: self.max_x = px
                    if py < self.min_y: self.min_y = py
                    if py > self.max_y: self.max_y = py

                    if id not in self.id_p_dict:
                        self.id_p_dict[id] = list()
                        self.id_v_dict[id] = list()
                        self.id_t_dict[id] = list()
                    self.id_p_dict[id].append([px, py])
                    self.id_v_dict[id].append([vx, vy])
                    self.id_t_dict[id].append(ts)
                    if ts not in self.t_p_dict:
                        self.t_p_dict[ts] = []
                        self.t_id_dict[ts] = []
                    self.t_p_dict[ts].append([px, py])
                    self.t_id_dict[ts].append(id)

        for pid in self.id_p_dict:
            self.id_p_dict[pid] = np.array(self.id_p_dict[pid])
            self.id_v_dict[pid] = np.array(self.id_v_dict[pid])
            self.id_t_dict[pid] = np.array(self.id_t_dict[pid])

        #  Find the time interval
        for pid, T in self.id_t_dict.items():
            if len(T) > 1:
                interval = int(round(T[1] - T[0]))
                if interval > 0:
                    self.interval = interval
                    break

    def get_all_trajs(self):
        all_trajs = []
        for key, val in sorted(self.id_p_dict.items()):
            all_trajs.append(val)
        return all_trajs

    def get_all_points(self):
        all_points = []
        for key, val in sorted(self.id_p_dict.items()):
            all_points.extend(val)
        return all_points

# def create_dataset(p_data, t_data, t_range, n_past=8, n_next=12):
#     dataset_t0 = []
#     dataset_x = []
#     dataset_y = []
#     for t in range(t_range.start, t_range.stop, 1):
#         for i in range(len(t_data)):
#             t0_ind = (np.where(t_data[i] == t))[0]
#             tP_ind = (np.where(t_data[i] == t - t_range.step * n_past))[0]
#             tF_ind = (np.where(t_data[i] == t + t_range.step * (n_next - 1)))[0]
#
#             if t0_ind.shape[0] == 0 or tP_ind.shape[0] == 0 or tF_ind.shape[0] == 0:
#                 continue
#
#             t0_ind = t0_ind[0]
#             tP_ind = tP_ind[0]
#             tF_ind = tF_ind[0]
#
#             dataset_t0.append(t)
#             dataset_x.append(p_data[i][tP_ind:t0_ind])
#             dataset_y.append(p_data[i][t0_ind:tF_ind + 1])
#
#
#     sub_batches = []
#     last_included_t = -1000
#     min_interval = 1
#     for i, t in enumerate(dataset_t0):
#         if t > last_included_t + min_interval:
#             sub_batches.append([i, i+1])
#             last_included_t = t
#
#         if t == last_included_t:
#             sub_batches[-1][1] = i + 1
#
#     sub_batches = np.array(sub_batches).astype(np.int16)
#     dataset_x_ = []
#     dataset_y_ = []
#     last_ind = 0
#     for ii, sb in enumerate(sub_batches):
#         dataset_x_.append(dataset_x[sb[0]:sb[1]])
#         dataset_y_.append(dataset_y[sb[0]:sb[1]])
#         sb[1] = sb[1] - sb[0] + last_ind
#         sb[0] = last_ind
#         last_ind = sb[1]
#
#     dataset_x = np.concatenate(dataset_x_)
#     dataset_y = np.concatenate(dataset_y_)
#
#     sub_batches = np.array(sub_batches).astype(np.int16)
#     dataset_x = np.array(dataset_x).astype(np.float32)
#     dataset_y = np.array(dataset_y).astype(np.float32)
#
#     return dataset_x, dataset_y, dataset_t0, sub_batches
#


if __name__ == '__main__':
    ParserETH('')   # TODO