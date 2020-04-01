import os
import sys
import numpy as np


class ParserSDD:
    """
        Parser class for SDD (Stanford Drone Dataset)
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation file: e.g. "OpenTraj/SDD/bookstore/video0/annotations.txt"

        Attributes:
            id_p_dict: map from pedestrian id to her trajectory (positions)
            id_v_dict: map from pedestrian id to her velocity data
            id_t_dict: map from pedestrian id to timestamps she appears
            t_id_dict: map from dataset timestamps to pedestrian ids
            t_p_dict : map from dataset timestamps to location of all pedestrians
            id_label_dict : map from object id to its label
                from {"Pedestrian", "Biker", "Skater", "Cart", "Car", "Bus"}
            min_t    : first timestamp
            max_t    : last timestamp
            interval : interval between timestamps
            [min_x, max_x], [min_y, max_y] : spacial extents of all the trajectories

    """

    def __init__(self, filename=''):
        self.actual_fps = 2.5
        self.id_p_dict = dict()
        self.id_v_dict = dict()
        self.id_t_dict = dict()
        self.id_label_dict = dict()
        self.t_id_dict = dict()
        self.t_p_dict = dict()
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 1
        self.min_x = 0
        self.min_y = 0
        self.max_x = 1920
        self.max_y = 1080
        if filename:
            self.load(filename)

    def load(self, filename, down_sample=1):
        self.id_p_dict = dict()
        self.id_t_dict = dict()
        self.id_label_dict = dict()
        self.t_p_dict = dict()

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
                    row = row.split(' ')
                    while '' in row: row.remove('')
                    if len(row) < 10: continue

                    id = round(float(row[0]))
                    ts = float(row[5])
                    if ts % down_sample != 0: continue

                    xl = round(float(row[1]))
                    yt = round(float(row[2]))
                    xr = round(float(row[3]))
                    yb = round(float(row[4]))
                    px = (xl + xr) / 2
                    py = (yt + yb) / 2

                    label = row[9].replace("\"", "").replace("\n", "")

                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts
                    if px < self.min_x: self.min_x = px
                    if px > self.max_x: self.max_x = px
                    if py < self.min_y: self.min_y = py
                    if py > self.max_y: self.max_y = py

                    if id not in self.id_p_dict:
                        self.id_p_dict[id] = list()
                        self.id_t_dict[id] = list()
                    self.id_p_dict[id].append([px, py])
                    self.id_t_dict[id].append(ts)
                    self.id_label_dict[id] = label
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


def count_objects(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))

    print(files)

    tot_Bikers = 0
    tot_Pedestrians = 0
    tot_Skateboarders = 0
    tot_Carts = 0
    tot_Cars = 0
    tot_Buss = 0
    for file in files:
        parser = ParserSDD()
        parser.load(file)

        Bikers = 0
        Pedestrians = 0
        Skateboarders = 0
        Carts = 0
        Cars = 0
        Buss = 0
        for label in parser.label_data:
            if label == "Biker":
                Bikers += 1
            elif label == "Pedestrian":
                Pedestrians += 1
            elif "Skateboard" in label:
                Skateboarders += 1
            elif label == "Cart":
                Carts += 1
            elif label == "Car":
                Cars += 1
            elif label == "Bus":
                Buss += 1

        tot_Bikers += Bikers
        tot_Pedestrians += Pedestrians
        tot_Skateboarders += Skateboarders
        tot_Carts += Carts
        tot_Cars += Cars
        tot_Buss += Buss

        print(file)
        print(Bikers, Pedestrians, Skateboarders, Carts, Cars, Buss)

    return tot_Bikers, tot_Pedestrians, tot_Skateboarders, tot_Carts, tot_Cars, tot_Buss


if __name__ == '__MAIN__':
    bikers, peds, skateboarders, carts, cars, buss = count_objects('../../SDD')
    print('final result')
    print(bikers, peds, skateboarders, carts, cars, buss)