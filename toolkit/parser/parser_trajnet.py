import os
import sys
import numpy as np
from .parser import TrajectoryDataset


class ParserTrajnet(TrajectoryDataset):
    """
        Parser class for TrajNet train and test files
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation file: e.g. "OpenTraj/ETH/seq_eth/obsmat.txt"
    """

    def __init__(self,  dataroot='', filter_files='', **kwargs):
        super().__init__()
        self.delimiter = ' '
        self.__ndim__ = 2
        if dataroot:
            self.__load__(dataroot, filter_files)

    def __load__(self, dataroot, filter_files=''):
        # search for files in a folder
        annotation_files = list()
        if not os.path.exists(dataroot):
            raise ValueError("No such file or directory: [%s]" % dataroot)
        elif not os.path.isdir(dataroot):
            annotation_files.append(dataroot)
            self.__title__ = os.path.splitext(os.path.basename(dataroot))[0]

        # TODO: check with regular expression, e.g.: `ETH | Zara01`
        else:  # a directory
            self.__title__ = 'trajnet'
            for root, dirs, files in os.walk(dataroot):
                for file in files:
                    if file.endswith(".txt"):
                        annotation_files.append(os.path.join(root, file))

        """
         when loading multiple annotation files, the timestamps might overlap
         That's why, we add an offset to timestamps as if they start after previous exp
        """
        timestamp_offset = 0

        for file in annotation_files:
            id_postfix = '-' + os.path.splitext(os.path.basename(file))[0]
            if 'stanford' in file:
                fps = 29.97
            elif 'biwi' in file or 'crowds' in file:
                fps = 25
            else:  # mot
                fps = 7
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                for i, line in enumerate(content):
                    row = line.split(self.delimiter)
                    while '' in row: row.remove('')
                    if len(row) < 4 or '?' in line: continue

                    ts = int(float(row[0])) + timestamp_offset
                    id = row[1] + id_postfix
                    px = float(row[2])
                    py = float(row[3])

                    if id not in self.__id_p_dict__:
                        self.__id_p_dict__[id] = list()
                        self.__id_t_dict__[id] = list()
                        self.__id_fps_dict__[id] = fps
                    # we assume the samples for each id are sorted by timestamp
                    self.__id_p_dict__[id].append(np.array([px, py]))
                    self.__id_t_dict__[id].append(ts)

            timestamp_offset = ts + 10
        self.__post_load__()


if __name__ == '__main__':
    # TODO: check argv[1]
    ParserTrajnet('')
