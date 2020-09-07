import pandas as pd
import numpy as np
import sys
import os
from crowdscan.crowd.trajdataset import TrajDataset


def loadChAOS(path, separator, **kwargs):
    traj = TrajDataset()
    # TODO
    #  ChAOS Style: 1 file per agent, pos_x, pos_y
    print("\nLoad Chaos style: not implemented yet\n")
    return traj

