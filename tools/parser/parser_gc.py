# @Authors : Javad Amirian
# @Email   : javad.amirian@inria.fr
# adopted from CIDNN: https://github.com/svip-lab/CIDNN/blob/master/tools/create_dataset.py

import numpy as np
import os


class ParserGC:
    def __init__(self):
        self.id_p_dict = dict()
        self.id_v_dict = dict()
        self.id_t_dict = dict()
        self.t_p_dict = dict()
        self.min_t = 0  # fixed
        self.max_t = 0
        self.interval = 1  # fixed
        self.GC_IMAGE_WIDTH = 1920
        self.GC_IMAGE_HEIGHT = 1080

    def load(self, filename):
        if '.npz' in filename:
            self.load_npz_file(filename)
        else:
            self.load_raw_data(filename)

    def load_npz_file(self, filename):
        pass

    def load_raw_data(self, raw_data_path):
        """
        create data from downloaded raw data to meta data ( a data structure to read easily)
        :param raw_data_path: downloaded raw data path
        :return:
        """

        dir_list = sorted(os.listdir(raw_data_path))
        p_num = len(dir_list)

        # + 1 because raw GC txt start from 1,just add a fake person whose pid = 0
        for ii in range(1, p_num + 1):
            self.id_p_dict[ii] = []
            self.id_v_dict[ii] = []
            self.id_t_dict[ii] = []

        # fill p_data
        self.max_t = 0
        for dir_name in dir_list:
            person_trajectory_txt_path = os.path.join(raw_data_path, dir_name)
            pid = int(dir_name.replace('.txt', ''))

            with open(person_trajectory_txt_path, 'r') as f:
                trajectory_list = f.read().split()
                for i in range(len(trajectory_list) // 3):
                    px = int(trajectory_list[3 * i]) / self.GC_IMAGE_WIDTH
                    py = int(trajectory_list[3 * i + 1]) / self.GC_IMAGE_HEIGHT
                    ts = int(trajectory_list[3 * i + 2]) // 20
                    self.max_t = max(self.max_t, ts)
                    self.id_p_dict[pid].append([px, py])
                    # self.id_v_dict[pid].append([vx, vy])
                    self.id_t_dict[pid].append(ts)
                    if ts not in self.t_p_dict:
                        self.t_p_dict[ts] = []
                    self.t_p_dict[ts].append([px, py])

        for pid in self.id_p_dict:
            print(pid)
            self.id_p_dict[pid] = np.array(self.id_p_dict[pid])
            self.id_t_dict[pid] = np.array(self.id_t_dict[pid])
            self.id_v_dict[pid] = self.id_p_dict[pid][1:] - self.id_p_dict[pid][:-1]
            if len(self.id_p_dict[pid]) == 1:
                self.id_v_dict[pid] = np.zeros((1, 2), dtype=np.float64)
            else:
                self.id_v_dict[pid] = np.append(self.id_v_dict[pid], self.id_v_dict[pid][-1].reshape(1,2), axis=0)

        # show some message
        print('pedestrian_data_list size: ', len(self.id_p_dict))

        # with open(meta_data_path, 'w') as f:
        #     json.dump({'frame_data_list': f_data_list, 'pedestrian_data_list': p_data_list}, f)
        # print('create %s successfully!' % meta_data_path)

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

    def create_dataset(self, data_path):
        # GC_meta_data_path = '../../data/GC_meta_data.json'
        # GC_train_test_data_path = '../../data/GC.npz'
        self.load(data_path)


if __name__ == '__main__':
    parser = ParserGC()
    parser.create_dataset('/home/cyrus/workspace2/OpenTraj/GC/Annotation')