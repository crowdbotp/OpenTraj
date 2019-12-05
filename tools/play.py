import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from parser.parser_eth import ParserETH
from parser.parser_sdd import ParserSDD


def to_image_frame(Hinv, loc):
    """
    Given H^-1 and world coordinates, returns (u, v) in image coordinates.
    """
    if loc.ndim > 1:
        locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
        loc_tr = np.transpose(locHomogenous)
        loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
        locXYZ = np.transpose(loc_tr/loc_tr[2])  # to pixels (from millimeters)
        return locXYZ[:, :2].astype(int)
    else:
        locHomogenous = np.hstack((loc, 1))
        locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
        locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
        return locXYZ[:2].astype(int)


def line_cv(im, ll, value, width):
    for tt in range(ll.shape[0] - 1):
        cv2.line(im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), value, width)


# ref_im_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/reference.png'
# traj_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt'
# homog_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/H.txt'
# parser = ParserETH()

ref_im_file = '/home/cyrus/workspace2/OpenTraj/SDD/bookstore/video0/reference.jpg'
traj_file = '/home/cyrus/workspace2/OpenTraj/SDD/bookstore/video0/annotations.txt'
homog_file = ''
parser = ParserSDD()

if os.path.exists(homog_file):
    Hinv = np.linalg.inv(np.loadtxt(homog_file))
else:
    Hinv = np.eye(3, 3)

parser.load(traj_file)
keys = sorted(parser.time_dict.keys())


if 'mp4' in ref_im_file:
    pass  # TODO: play with video
elif os.path.exists(ref_im_file):
    ref_im = cv2.imread(ref_im_file)

    for frame_id in keys:
        print(frame_id)
        id_ps = parser.time_dict[frame_id]
        ref_im_copy = np.copy(ref_im)
        for id_xy_pair in id_ps:
            id = id_xy_pair[0]
            cur_xy = np.array(id_xy_pair[1])
            cur_UV = to_image_frame(Hinv, cur_xy)

            # fetch entire trajectory
            p_id = parser.id_p_dict[id]
            P_id = to_image_frame(Hinv, p_id)
            line_cv(ref_im_copy, P_id, (255, 255, 0), 2)
            cv2.circle(ref_im_copy, (cur_UV[1], cur_UV[0]), 5, (0, 0, 255), 2)

        cv2.imshow('OpenTraj', ref_im_copy)
        cv2.waitKey(50)

