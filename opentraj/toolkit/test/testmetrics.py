from toolkit.loaders.loader_eth import load_eth
from toolkit.benchmarking.metrics.individual import motion
from toolkit.benchmarking.metrics.individual import path_length
from toolkit.benchmarking.metrics.agent_to_agent import pcf, distance
import matplotlib.pyplot as plt
import numpy as np


def run(module_directory, args):

    print("\n\n-----------------------------\nRunning test metrics\n-----------------------------")

    eth_dataset = load_eth(module_directory + '/tests/toy trajectories/ETH/seq_eth/obsmat.txt',
                           args.separator)
    all_trajs = eth_dataset.get_trajectories()

    for traj in all_trajs:
        speed = motion.speed(traj)
        p_len = path_length.path_length(traj)
        p_eff = path_length.path_efficiency(traj)

    all_frames = eth_dataset.get_frames()

    # calc pcf on a dataset
    pcf_accum = []
    pcf_range = np.arange(0.2,  # starting radius
                          8,    # end radius
                          0.2)  # step size

    for frame in all_frames:
        pcf_values_t = pcf.pcf(frame[['pos_x', 'pos_y']],
                               list(pcf_range), sigma=0.25)
        if not len(pcf_accum):
            pcf_accum = pcf_values_t
        else:
            pcf_accum += pcf_values_t
    avg_pcf = pcf_accum / len(all_frames)
    print('average pcf = ', avg_pcf)

    # social spaces
    social_spaces = distance.social_space(eth_dataset.data)
    print(social_spaces.values())

    plt.hist(social_spaces.values(), bins=np.arange(0, 5, 0.2), density=True)
    plt.show()

    # traj.plot(y='acc x')
    # ax = plt.gca()
    # for data in np.unique(traj.index.get_level_values('agent id')):
    #     random_color = np.random.rand(3,)
    #     traj[traj.index.get_level_values('agent id') == data].plot(y='acc x', c = random_color, ax=ax)
    #
    # plt.show()
    # print(np.unique(traj.index.get_level_values('agent id')))
    print("\n\n-----------------------------\nTest metrics done\n-----------------------------")

