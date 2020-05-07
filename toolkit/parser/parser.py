import sys


class TrajectoryDataset:
    """
        Parser class for Hermes experiments
        -------
        You can either use the class constructor or call 'load' method,
        by passing the annotation folder: e.g. "OpenTraj/HERMES/Bottleneck_Data/uo-180-070.txt"

        Attributes:
            t_p_dict : map from a timestamp to all agent's locations at t
            t_v_dict : map from a timestamp to all agent's velocities at t
            t_id_dict: map from a timestamp to all agent ids at t
            id_t_dict: map from an agent id to all timestamps she appears
            id_p_dict: map from an agent id to all her locations (trajectory)
            id_v_dict: map from an agent id to all her velocity samples
            id_g_dict: map from an agent id to ids of her groupmates (if groups are annotated)
            id_l_dict: map from an agent id to her class label (e.g. Pedestrian, cyclist, car ...)
            min_t    : first timestamp
            max_t    : last timestamp
            interval : interval between timestamps
            [min_x, max_x], [min_y, max_y] : spacial extents of all the trajectories
    """

    # TODO: merge p & v -> x
    # TODO: rename dict to map
    #
    def __init__(self, filename='', **kwargs):
        self.t_p_dict  = dict()
        self.t_v_dict  = dict()
        self.t_id_dict = dict()
        self.id_t_dict = dict()
        self.id_p_dict = dict()
        self.id_v_dict = dict()
        self.id_g_dict = dict()
        self.id_l_dict = dict()
        self.dataset_name = ''
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = -1
        self.min_x = 1000
        self.min_y = 1000
        self.max_x = 0
        self.max_y = 0

    def all_ids(self):
        pass

    def all_trajectories(self):
        all_trajs = []
        for key, val in sorted(self.id_p_dict.items()):
            all_trajs.append(val)
        return all_trajs

    def all_samples(self):
        all_samples = []
        for key, val in sorted(self.id_p_dict.items()):
            all_samples.extend(val)
        return all_samples

    def create_dataset(self, n_obsv, n_pred, **kwargs):
        pass

    def num_agents(self):
        return len(self.id_p_dict)


if __name__ == '__main__':
    class ParserTest(TrajectoryDataset):
        def foo(self, **kwargs):
            print(self.id_t_dict)
            for key, value in kwargs.items():
                print("%s == %s" % (key, value))

    ParserTest('').foo(x=2)
