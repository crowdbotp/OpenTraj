# @Authors : Javad Amirian
# @Email   : javad.amirian@inria.fr

import numpy as np
import os


class ParserHermes:
    """
            Parser class for Hermes experiments
            -------
            You can either use the class constructor or call 'load' method,
            by passing the annotation folder: e.g. "OpenTraj/HERMES/Bottleneck_Data/uo-180-070.txt"

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
        self.max_t = 0
        self.min_t = 0  # fixed
        self.interval = 20  # fixed
        self.min_x = 0
        self.min_y = 0
        self.max_x = 1920
        self.max_y = 1080
        if filename:
            self.load(filename)

    def load(self, filename, down_sample=1, delimit=' '):
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
                content = data_file.readlines()
                for i, row in enumerate(content):
                    row = row.split(delimit)
                    while '' in row: row.remove('')
                    if len(row) < 5: continue

                    id = int(row[0])
                    ts = int(row[1])
                    if ts % down_sample != 0:
                        continue

                    px = float(row[2])/100.
                    py = float(row[4])/100.

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
                    self.id_t_dict[id].append(ts)

                    if ts not in self.t_p_dict:
                        self.t_p_dict[ts] = []
                        self.t_id_dict[ts] = []
                    self.t_p_dict[ts].append([px, py])
                    self.t_id_dict[ts].append(id)

            for pid in self.id_p_dict:
                self.id_p_dict[pid] = np.array(self.id_p_dict[pid])
                self.id_t_dict[pid] = np.array(self.id_t_dict[pid])
                self.id_v_dict[pid] = self.id_p_dict[pid][1:] - self.id_p_dict[pid][:-1]
                if len(self.id_p_dict[pid]) == 1:
                    self.id_v_dict[pid] = np.zeros((1, 2), dtype=np.float64)
                else:
                    self.id_v_dict[pid] = np.append(self.id_v_dict[pid], self.id_v_dict[pid][-1].reshape(1, 2), axis=0)

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


if __name__ == '__main__':
    parser = ParserHermes("../../HERMES/Bottleneck_Data/uo-180-070.txt")
    n_ped = len(parser.id_p_dict.items())
    if n_ped:
        print("HermesParser successfully loaded file and found %d pedestrians" % n_ped)
    else:
        print("HermesParser failed loading file(s)")