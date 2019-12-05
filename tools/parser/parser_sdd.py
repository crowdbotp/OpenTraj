import os
import sys
import numpy as np
from builtins import ValueError


class ParserSDD:
    def __init__(self):
        self.actual_fps = 2.5
        self.id_p_dict = dict()
        self.id_t_dict = dict()
        self.id_label_dict = dict()
        self.time_dict = dict()
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = 1

    def load(self, filename, down_sample=1):
        self.id_p_dict = dict()
        self.id_t_dict = dict()
        self.id_label_dict = dict()
        self.time_dict = dict()

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

                    xmin = round(float(row[1]))
                    ymin = round(float(row[2]))
                    xmax = round(float(row[3]))
                    ymax = round(float(row[4]))
                    px = (xmin + xmax) / 2
                    py = (ymin + ymax) / 2

                    label = row[9].replace("\"", "").replace("\n", "")

                    if ts < self.min_t: self.min_t = ts
                    if ts > self.max_t: self.max_t = ts

                    if id not in self.id_p_dict:
                        self.id_p_dict[id] = list()
                        self.id_t_dict[id] = list()
                    self.id_p_dict[id].append([px, py])
                    self.id_t_dict[id].append(ts)
                    self.id_label_dict[id] = label
                    if ts not in self.time_dict:
                        self.time_dict[ts] = []
                    self.time_dict[ts].append((id, [px, py]))

        for id_ in self.id_p_dict:
            self.id_p_dict[id_] = np.array(self.id_p_dict[id_])
            self.id_t_dict[id_] = np.array(self.id_t_dict[id_])


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