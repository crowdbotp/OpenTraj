# Author: Javad Amirian
# Email: amiryan.j@gmail.com

"""
Constant-Velocity Prediction Baseline
"""

import numpy as np


def const_vel(x_t, pred_dts):
    """
    :param x_t: 4d vector = current state (px, py, vx, vy)
    :param pred_dts: list of T delta_t for doing prediction (e.g. [0.2s, 0.4s])
    :return: T x 2 array = predicted positions
    """
    pt = x_t[:2]
    vt = x_t[2:]
    preds = np.array(pred_dts).reshape((-1, 1)) * vt.reshape((1, 2)) + pt
    return preds
