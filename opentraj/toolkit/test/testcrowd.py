from toolkit.core.trajdataset import TrajDataset
from random import random

def run():

    traj_dataset = TrajDataset()
    traj_dataset.add_agent(12, 10, 1., 0., 125.)

    print("\n\n-----------------------------\nRunning test crowd\n-----------------------------")

    # f0 = frame(0)
    #
    # print("\n1:\n",f0.get_frame())
    #
    # f0.add_agent(12,1,0,125)
    # f0.add_agent(15,10,52,15)
    #
    # print("\n2:\n",f0.get_frame())
    #
    # f0.set_agents_list_size(1)
    # f0.add_agent(132,15,25,1225)
    #
    # print("\n3:\n",f0.get_frame())
    #
    # f0.reset_frame()
    #
    # print("\n4:\n",f0.get_frame())
    #
    # t = TrajectoryDataset()
    #
    # for i in range(0,3):
    #     f = frame(i+5)
    #     f.add_agent(15,random(),random(),random())
    #     f.add_agent(17,random(),random(),random())
    #     t.add_frame(f)
    #
    # traj = t.get_trajectories()
    # print("Trajectory:\n1:\n",traj)

    # t.reset_trajectory()
    # traj = t.get_trajectories()
    # print("\n3: reset \n",traj)


    print("\n\n-----------------------------\nTest crowd done\n-----------------------------")

