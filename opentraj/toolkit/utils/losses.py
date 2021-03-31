# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import numpy as np


def euclidean_dist(p1, p2):
    """:return euclidean distance between two 2D points p1 and p2"""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def average_displacement(tr_pred, tr_true):
    """
    Average displacement error between two trajlets
    :param tr_pred: prediction trajlet [NxTx2]
    :param tr_true: ground truth trajlet [NxTx2]
    :return: (a non-negative float number)
    """
    # N x T x 2

    total_dist = 0
    for i in range(len(tr_true)):
        total_dist += euclidean_dist(tr_pred[i], tr_true[i])
    return total_dist/len(tr_true)


def final_displacement(tr_pred, tr_true):
    """
    Final displacement error between two trajlets
    :param tr_pred: prediction trajlet [NxTx2]
    :param tr_true: ground truth trajlet [NxTx2]
    :return: (a non-negative float number)
    """
    return euclidean_dist(tr_pred[-1], tr_true[-1])


