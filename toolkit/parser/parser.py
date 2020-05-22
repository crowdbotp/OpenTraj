import sys
import numpy as np


class TrajectoryDataset:
    """
        Parser class for Hermes experiments
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation folder: e.g. "OpenTraj/HERMES/Bottleneck_Data/uo-180-070.txt"

        Attributes:
            t_p_dict : map from a timestamp to actual time (considering fps)
            t_p_dict : map from a timestamp to all agent's locations at t
            t_v_dict : map from a timestamp to all agent's velocities at t
            t_id_dict: map from a timestamp to all agent ids at t
            id_t_dict: map from an agent id to all timestamps she appears
            id_p_dict: map from an agent id to all her locations (trajectory)
            id_v_dict: map from an agent id to all her velocity samples
            id_a_dict: map from an agent id to all her acceleration samples
            id_g_dict: map from an agent id to ids of her groupmates (if groups are annotated)
            id_l_dict: map from an agent id to her class label (e.g. Pedestrian, cyclist, car ...)
            min_t    : first timestamp
            max_t    : last timestamp
            interval : interval between timestamps
            [[min_x, min_y, ...], [max_x, max_y, ...] : spacial extents of all the trajectories
    """

    # TODO: merge p & v -> x ?
    # TODO: add functions
    def __init__(self, dataroot='', filter_files='', **kwargs):
        self.__t_t_dict__  = dict()
        self.__t_p_dict__  = dict()
        self.__t_v_dict__  = dict()
        self.__t_id_dict__ = dict()
        self.__id_t_dict__ = dict()
        self.__id_p_dict__ = dict()
        self.__id_v_dict__ = dict()
        self.__id_a_dict__ = dict()
        self.__id_g_dict__ = dict()  # list of group-mates
        self.__id_l_dict__ = dict()  # label of agent
        self.__id_fps_dict__ = dict()  # fps value, this trajectory is annotated with
        self.__title__ = ''
        self.__ndim__ = 0
        # self.__interval__ = # deprecated => use (__id_t_dict__[pid][1] - __id_t_dict__[pid][0])
        self.__fps__ = 25   # default value => FIXME: should be updated by parser
        # self.dt = lambda: self.__interval__ / self.__fps__
        self.__min_t__ = int(sys.maxsize)
        self.__max_t__ = -1
        # TODO: refactor this part
        self.min_x = sys.maxsize
        self.min_y = sys.maxsize
        self.min_z = sys.maxsize
        self.max_x = -sys.maxsize
        self.max_y = -sys.maxsize
        self.max_z = -sys.maxsize

    def erase(self):
        self.__init__()

    # should be reimplemented
    def __load__(self, dataroot, filter_files='') -> None:
        pass

    def __post_load__(self):
        if not self.__id_t_dict__:
            print("Error! The dataset is empty")
            return -1

        #  Find the time interval
        # self.__interval__ = self.__id_t_dict__.values()[0][1] - self.__id_t_dict__.values()[0][0]
        # for pid, T in self.__id_t_dict__.items():
        #     if len(T) > 1:
        #         interval = int(round(T[1] - T[0]))
        #         if interval > 0:
        #             self.__interval__ = interval
        #             break

        for pid in self.__id_p_dict__.keys():
            # if labels are not assigned
            if pid not in self.__id_l_dict__:
                self.__id_l_dict__[pid] = "unknown"  # default agent type
            # if group-mates are not assigned
            if pid not in self.__id_g_dict__:
                self.__id_g_dict__[pid] = []  # no group-mate
            # if video fps is not set
            if pid not in self.__id_fps_dict__:
                self.__id_fps_dict__[pid] = 30.  # default video fps!!!

        for pid in self.__id_p_dict__.keys():
            # make sure data are stored in numpy arrays, not in lists
            self.__id_t_dict__[pid] = np.array(self.__id_t_dict__[pid])
            self.__id_p_dict__[pid] = np.array(self.__id_p_dict__[pid])
            # ====================

            if pid in self.__id_v_dict__:
                self.__id_v_dict__[pid] = np.array(self.__id_v_dict__[pid])
            else:
                self.__id_v_dict__[pid] = np.zeros_like(self.__id_p_dict__[pid])
                if len(self.__id_p_dict__[pid]) > 1:
                    dt = np.diff(self.__id_t_dict__[pid]) / self.__id_fps_dict__[pid]
                    self.__id_v_dict__[pid][:-1] = np.diff(self.__id_p_dict__[pid], axis=0) / dt[:, None]
                    self.__id_v_dict__[pid][-1] = self.__id_v_dict__[pid][-2]

            if pid in self.__id_a_dict__:
                self.__id_a_dict__[pid] = np.array(self.__id_a_dict__[pid])
            else:
                self.__id_a_dict__[pid] = np.zeros_like(self.__id_p_dict__[pid])
                if len(self.__id_p_dict__[pid]) > 1:
                    dt = np.diff(self.__id_t_dict__[pid]) / self.__id_fps_dict__[pid]
                    self.__id_a_dict__[pid][1:] = np.diff(self.__id_v_dict__[pid], axis=0) / dt[:, None]
                    self.__id_a_dict__[pid][0] = self.__id_a_dict__[pid][1]

            for t_ind, ts in enumerate(self.__id_t_dict__[pid]):
                if ts not in self.__t_id_dict__:
                    self.__t_id_dict__[ts] = []
                    self.__t_p_dict__[ts] = []
                    self.__t_v_dict__[ts] = []
                self.__t_id_dict__[ts].append(pid)
                self.__t_p_dict__[ts].append(self.__id_p_dict__[pid][t_ind])
                self.__t_v_dict__[ts].append(self.__id_v_dict__[pid][t_ind])

        self.min_t = min(self.__t_id_dict__.keys())
        self.max_t = max(self.__t_id_dict__.keys())

        all_samples = np.array(self.get_samples())
        self.min_x = min(all_samples[:, 0])
        self.max_x = max(all_samples[:, 0])
        self.min_y = min(all_samples[:, 1])
        self.max_y = max(all_samples[:, 1])
        if self.__ndim__ > 2:
            self.min_z = min(all_samples[:, 2])
            self.max_z = max(all_samples[:, 2])

    def get_all_ids(self):
        return sorted(self.__id_p_dict__.keys())

    # TODO: take min_t=[-1], max_t=[sys.maxsize] as args
    # FIXME: do we need to ask ?
    def get_trajectories(self, ids=[], include_vel=False, include_acc=False):
        if not ids:
            ids = sorted(self.__id_p_dict__.keys())

        positions = [traj for id, traj in self.__id_p_dict__.items() if id in ids]
        if not include_vel:
            trajectories = positions
        elif not include_acc:
            velocities = [vel for id, vel in self.__id_v_dict__.items() if id in ids]
            trajectories = [np.stack([positions[i], velocities[i]])
                            for i in range(len(ids))]
        else:  # if include_acc:
            velocities = [vel for id, vel in self.__id_v_dict__.items() if id in ids]
            accelerations = [acc for id, acc in self.__id_a_dict__.items() if id in ids]
            trajectories = [np.stack([positions[i], velocities[i], accelerations[i]])
                            for i in range(len(ids))]
        return trajectories

    # TODO: take min_t=[-1], max_t=[sys.maxsize] as args
    def get_samples(self, ids=[], include_vel=False, include_acc=False):
        if not ids:
            ids = sorted(self.__id_p_dict__.keys())

        samples = []
        for id, p in sorted(self.__id_p_dict__.items()):
            if id not in ids: continue
            if not include_vel:
                samples.extend(p)
            elif not include_acc:
                v = self.__id_v_dict__[id]
                samples.extend(np.concatenate([p, v], axis=1))
            else:  # if include_acc:
                v = self.__id_v_dict__[id]
                a = self.__id_a_dict__[id]
                samples.extend(np.concatenate([p, v, a], axis=1))

        return samples

    def get_agent_timestamps(self, ids=[], include_vel=False):
        if not ids:
            ids = sorted(self.__id_p_dict__.keys())
        trajectories = [traj for id, traj in self.__id_p_dict__.items() if id in ids]

        # FIXME: do we need to return (p|v) ?
        if include_vel:
            velocities = [vel for id, vel in self.__id_v_dict__ if id in ids]
            trajectories = [np.stack([trajectories[i], velocities[i]])
                            for i in len(ids)]

        return trajectories

    # TODO: to be implemented
    def create_prediction_dataset(self, n_obsv, n_pred, **kwargs):
        pass

    def num_agents(self):
        return len(self.__id_p_dict__)

    def data_dimension(self):
        return self.__ndim__

if __name__ == '__main__':
    class ParserTest(TrajectoryDataset):
        def foo(self, **kwargs):
            print(self.__id_t_dict__)
            for key, value in kwargs.items():
                print("%s == %s" % (key, value))

    ParserTest('').foo(x=2)
