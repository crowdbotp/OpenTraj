import os
import sys
import numpy as np
from .parser import TrajectoryDataset


class ParserETH(TrajectoryDataset):
    """
        Parser class for ETH dataset and UCY dataset
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation file: e.g. "OpenTraj/ETH/seq_eth/obsmat.txt"
    """

    def __init__(self, dataroot='', filter_files='', **kwargs):
        super().__init__()
        self.delimiter = ' '
        self.__ndim__ = 2
        # groups_filename = ''  # FIXME
        if dataroot:
            self.__load__(dataroot)

    def __load__(self, dataroot, down_sample=1, delimit=' ', groups_filename=''):
        self.dataset_name = dataroot
        # to search for files in a folder?
        annotation_files = list()
        if not os.path.exists(dataroot):
            raise ValueError("No such file or directory: [%s]" % dataroot)
        elif not os.path.isdir(dataroot):
            annotation_files.append(dataroot)
            self.dataset_name = os.path.splitext(os.path.basename(dataroot.replace('/obsmat.txt','')))[0]

        # TODO: check with regular expression, e.g.: `ETH | Zara01`
        else:  # a directory
            self.dataset_name = 'ETH-UCY'
            for root, dirs, files in os.walk(dataroot):
                for file in files:
                    if file.endswith(".txt"):
                        annotation_files.append(os.path.join(root, file))

        timestamp_offset = 0
        fps = 16    #  fixed

        for file in annotation_files:
            id_postfix = '-' + os.path.splitext(os.path.basename(dataroot.replace('/obsmat.txt','')))[0]
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                for i, row in enumerate(content):
                    row = row.split(delimit)
                    while '' in row: row.remove('')
                    if len(row) < 8: continue

                    ts = int(float(row[0])) + timestamp_offset
                    id = row[1] + id_postfix
                    if ts % down_sample != 0:
                        continue

                    px = float(row[2])
                    py = float(row[4])
                    vx = float(row[5])
                    vy = float(row[7])

                    if id not in self.__id_p_dict__:
                        self.__id_p_dict__[id] = list()
                        self.__id_v_dict__[id] = list()
                        self.__id_t_dict__[id] = list()
                        self.__id_fps_dict__[id] = fps
                    self.__id_p_dict__[id].append([px, py])
                    self.__id_v_dict__[id].append([vx, vy])
                    self.__id_t_dict__[id].append(ts)

            timestamp_offset = ts + 10

        # FIXME:  for each dir you should pick the corresponding groups.txt
        if os.path.exists(groups_filename):
            with open(groups_filename, 'r') as group_file:
                content = group_file.read().splitlines()
                for row in content:
                    row = list(filter(None, row.split(' ')))
                    ids = [int(id) for id in row]
                    for id in ids:
                        self.__id_g_dict__[id] = ids.copy()
                        self.__id_g_dict__[id].remove(id)

        self.__post_load__()


if __name__ == '__main__':
    ParserETH('')   # TODO