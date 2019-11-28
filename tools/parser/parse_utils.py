import csv
import math
import sys
from builtins import ValueError

import numpy as np
import os
# from pandas import DataFrame, concat


class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy + self.min_y * shift
        else:
            return False

        return data_copy


class TrajnetParser:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.actual_fps = 0.
        self.delimit = ' '
        self.p_data = []
        self.v_data = []
        self.t_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 6

    def load(self, filename, down_sample=1):
        pos_data_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()

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
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                id_list = list()
                for i, row in enumerate(content):
                    row = row.split(self.delimit)
                    while '' in row: row.remove('')
                    if len(row) < 4: continue

                    ts = float(row[0])
                    id = round(float(row[1]))
                    if ts % down_sample != 0:
                        continue
                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts

                    px = float(row[2])
                    py = float(row[3])

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                    pos_data_dict[id].append(np.array([px, py]))
                    time_data_dict[id] = np.hstack((time_data_dict[id], np.array([ts])))
            self.all_ids += id_list

        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            self.t_data.append(np.array(time_data_dict[key]))

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()


class SDD_Parsrer:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.actual_fps = 0.
        self.delimit = ' '
        self.p_data = []
        self.t_data = []
        self.label_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 12

    def load(self, filename, down_sample=12):
        pos_data_dict = dict()
        vel_data_dict = dict()
        label_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()

        if 'zara' in filename:
            self.delimit = '\t'

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

        self.actual_fps = 2.5
        for file in file_names:
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                id_list = list()
                for i, row in enumerate(content):
                    row = row.split(self.delimit)
                    while '' in row: row.remove('')
                    if len(row) < 10: continue

                    id = round(float(row[0]))
                    ts = float(row[5])
                    if ts % down_sample != 0: continue

                    xmin = round(float(row[1]))
                    ymin = round(float(row[2]))
                    xmax = round(float(row[3]))
                    ymax = round(float(row[4]))

                    label = row[9]
                    label = label.replace("\n", "")
                    label = label.replace("\"", "")

                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts

                    px = (xmin + xmax) / 2
                    py = (ymin + ymax) / 2

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                        label_dict[id] = label
                        last_t = ts
                    pos_data_dict[id].append(np.array([px, py]))
                    time_data_dict[id] = np.hstack((time_data_dict[id], np.array([ts])))
            self.all_ids += id_list

        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            self.t_data.append(np.array(time_data_dict[key]).astype(np.int32))
            self.label_data.append(label_dict[key])

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()


class BIWIParser:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.delimit = ' '
        self.p_data = []
        self.v_data = []
        self.t_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = -1

    def load(self, filename, down_sample=1):
        pos_data_dict = dict()
        vel_data_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()

        if 'zara' in filename:
            self.delimit = '\t'

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
                id_list = list()
                for i, row in enumerate(content):
                    row = row.split(self.delimit)
                    while '' in row: row.remove('')
                    if len(row) < 8: continue

                    ts = float(row[0])
                    id = round(float(row[1]))
                    if ts % down_sample != 0:
                        continue
                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts


                    px = float(row[2])
                    py = float(row[4])
                    vx = float(row[5])
                    vy = float(row[7])

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        vel_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                        last_t = ts
                    pos_data_dict[id].append(np.array([px, py]))
                    vel_data_dict[id].append(np.array([vx, vy]))
                    time_data_dict[id] = np.hstack((time_data_dict[id], np.array([ts])))
            self.all_ids += id_list

        for ped_id, ped_T in time_data_dict.items():
            if len(ped_T) > 1:
                interval = int(round(ped_T[1] - ped_T[0]))
                if interval > 0:
                    self.interval = interval
                    break

        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            # TODO: you can apply a Kalman filter/smoother on v_data
            vels_i = np.array(vel_data_dict[key])
            self.v_data.append(vels_i)
            self.t_data.append(np.array(time_data_dict[key]).astype(np.int32))

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()


class SeyfriedParser:
    def __init__(self):
        self.scale = Scale()
        self.actual_fps = 0.

    def load(self, filename, down_sample=4):
        '''
        Loads datas of seyfried experiments
        * seyfried template:
        >> n_Obstacles
        >> x1[i] y1[i] x2[i] y2[i] x(n_Obstacles)
        >> fps
        >> id, timestamp, pos_x, pos_y, pos_z
        >> ...
        :param filename: dataset file with seyfried template
        :param down_sample: To take just one sample every down_sample
        :return:
        '''
        pos_data_list = list()
        vel_data_list = list()
        time_data_list = list()

        # check to search for many files?
        file_names = list()
        if '*' in filename:
            files_path = filename[:filename.index('*')]
            extension = filename[filename.index('*')+1:]
            for file in os.listdir(files_path):
                if file.endswith(extension):
                    file_names.append(files_path+file)
        else:
            file_names.append(filename)

        for file in file_names:
            with open(file, 'r') as data_file:
                csv_reader = csv.reader(data_file, delimiter=' ')
                id_list = list()
                i = 0
                for row in csv_reader:
                    i += 1
                    if i == 4:
                        fps = float(row[0])
                        self.actual_fps = fps / down_sample

                    if len(row) != 5:
                        continue

                    id = row[0]
                    ts = float(row[1])
                    if ts % down_sample != 0:
                        continue

                    px = float(row[2])/100.
                    py = float(row[3])/100.
                    pz = float(row[4])/100.
                    if id not in id_list:
                        id_list.append(id)
                        pos_data_list.append(list())
                        vel_data_list.append(list())
                        time_data_list.append(np.empty((0), dtype=int))
                        last_px = px
                        last_py = py
                        last_t = ts
                    pos_data_list[-1].append(np.array([px, py]))
                    v = np.array([px - last_px, py - last_py]) * fps / (ts - last_t + np.finfo(float).eps)
                    vel_data_list[-1].append(v)
                    time_data_list[-1] = np.hstack((time_data_list[-1], np.array([ts])))

        p_data = list()
        v_data = list()

        for i in range(len(pos_data_list)):
            poss_i = np.array(pos_data_list[i])
            p_data.append(poss_i)
            # TODO: you can apply a Kalman filter/smoother on v_data
            vels_i = np.array(vel_data_list[i])
            v_data.append(vels_i)
        t_data = np.array(time_data_list)

        for i in range(len(pos_data_list)):
            poss_i = np.array(pos_data_list[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()

        return p_data, v_data, t_data

#
# def to_supervised(data, n_in=1, n_out=1, diff_in=False, diff_out=True, drop_nan=True):
#     '''
#     @CopyRight: Code is inspired by weblog of machinelearningmastery.com
#     Copies the data columns (of an nD sequence) so that for each timestep you have a "in" seq and an "out" seq
#     :param data:
#     :param n_in: length of "in" seq (number of observations)
#     :param n_out: length of "out" seq (number of predictions)
#     :param diff_in: if True the "in" columns are differential otherwise will be absolute
#     :param diff_out: if True the "out" columns are differential otherwise will be absolute
#     :param drop_nan: if True eliminate the samples that contains nan (due to shift operation)
#     :return: a table whose columns are n_in * nD (observations) and then n_out * nD (predictions)
#     '''
#
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = DataFrame(data)
#     cols, names = list(), list()
#
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         names += [('var_in%d(t-%d)' % (j + 1, i-1)) for j in range(n_vars)]
#         if diff_in:
#             cols.append(df.shift(i-1) - df.shift(i))
#         else:
#             cols.append(df.shift(i-1))
#
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(1, n_out+1):
#         names += [('var_out%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#         if diff_out:
#             cols.append(df.shift(-i) - df.shift(0))  # displacement
#         else:
#             cols.append(df.shift(-i))  # position
#
#     # put it all together
#     agg = concat(cols, axis=1)
#     agg.columns = names
#
#     # drop rows with NaN values
#     if drop_nan:
#         agg.dropna(inplace=True)
#
#     return agg.values


def create_dataset(p_data, t_data, t_range, n_past=8, n_next=12):
    dataset_t0 = []
    dataset_x = []
    dataset_y = []
    for t in range(t_range.start, t_range.stop, 1):
        for i in range(len(t_data)):
            t0_ind = (np.where(t_data[i] == t))[0]
            tP_ind = (np.where(t_data[i] == t - t_range.step * n_past))[0]
            tF_ind = (np.where(t_data[i] == t + t_range.step * (n_next - 1)))[0]

            if t0_ind.shape[0] == 0 or tP_ind.shape[0] == 0 or tF_ind.shape[0] == 0:
                continue

            t0_ind = t0_ind[0]
            tP_ind = tP_ind[0]
            tF_ind = tF_ind[0]

            dataset_t0.append(t)
            dataset_x.append(p_data[i][tP_ind:t0_ind])
            dataset_y.append(p_data[i][t0_ind:tF_ind + 1])


    sub_batches = []
    last_included_t = -1000
    min_interval = 1
    for i, t in enumerate(dataset_t0):
        if t > last_included_t + min_interval:
            sub_batches.append([i, i+1])
            last_included_t = t

        if t == last_included_t:
            sub_batches[-1][1] = i + 1

    sub_batches = np.array(sub_batches).astype(np.int16)
    dataset_x_ = []
    dataset_y_ = []
    last_ind = 0
    for ii, sb in enumerate(sub_batches):
        dataset_x_.append(dataset_x[sb[0]:sb[1]])
        dataset_y_.append(dataset_y[sb[0]:sb[1]])
        sb[1] = sb[1] - sb[0] + last_ind
        sb[0] = last_ind
        last_ind = sb[1]

    dataset_x = np.concatenate(dataset_x_)
    dataset_y = np.concatenate(dataset_y_)

    sub_batches = np.array(sub_batches).astype(np.int16)
    dataset_x = np.array(dataset_x).astype(np.float32)
    dataset_y = np.array(dataset_y).astype(np.float32)

    return dataset_x, dataset_y, dataset_t0, sub_batches

