import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from parser.parser_eth import ParserETH
from parser.parser_sdd import ParserSDD
from parser.parser_gc  import ParserGC


def is_a_video(filename):
    return '.mp4' in media_file or '.avi' in media_file


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


def play(parser, Hinv, media_file):
    keys = sorted(parser.t_p_dict.keys())

    if os.path.exists(media_file):
        if is_a_video(media_file):
            cap = cv2.VideoCapture(media_file)
        else:
            ref_im = cv2.imread(media_file)

    for frame_id in range(keys[0], keys[-1]):
        print(frame_id)
        if is_a_video(media_file):
            ret, ref_im = cap.read()
            pass  # TODO: play with video
        # print(xys_t)
        ref_im_copy = np.copy(ref_im)
        if frame_id in keys:
            xys_t = parser.t_p_dict[frame_id]
            ids_t = parser.t_id_dict[frame_id]
        for ii, id in enumerate(ids_t):
            cur_xy = np.array(xys_t[ii])
            cur_UV = to_image_frame(Hinv, cur_xy)
            # cur_UV[1] = 720 + cur_UV[1]
            # fetch entire trajectory
            traj_id = parser.id_p_dict[id]
            TRAJ_id = to_image_frame(Hinv, traj_id)
            line_cv(ref_im_copy, TRAJ_id, (255, 255, 0), 2)
            cv2.circle(ref_im_copy, (cur_UV[1], cur_UV[0]), 5, (0, 0, 255), 2)

        cv2.imshow('OpenTraj', ref_im_copy)
        key = cv2.waitKey(100) & 0xFF
        if key == 27:
            break


if __name__ == '__main__':
    parser = ParserETH()
    # annot_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/obsmat.txt'
    # media_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/reference.png'
    # homog_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_eth/H.txt'

    # annot_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_hotel/obsmat.txt'
    # homog_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_hotel/H.txt'
    # media_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_hotel/reference.png'
    # media_file = '/home/cyrus/workspace2/OpenTraj/ETH/seq_hotel/video.avi'

    annot_file = '/home/cyrus/workspace2/OpenTraj/UCY/data_zara02/obsmat.txt'
    homog_file = '/home/cyrus/workspace2/OpenTraj/UCY/data_zara02/H.txt'
    media_file = '/home/cyrus/workspace2/OpenTraj/UCY/data_zara02/reference.png'
    media_file = '/home/cyrus/workspace2/OpenTraj/UCY/data_zara02/video.avi'

    # annot_file = '/home/cyrus/workspace2/OpenTraj/UCY/st3_dataset/obsmat_px.txt'
    # media_file = '/home/cyrus/workspace2/OpenTraj/UCY/st3_dataset/reference.png'
    # homog_file = '/home/cyrus/workspace2/OpenTraj/UCY/st3_dataset/H_iw.txt'

    # parser = ParserSDD()
    # annot_file = '/home/cyrus/workspace2/OpenTraj/SDD/bookstore/video0/annotations.txt'
    # media_file = '/home/cyrus/workspace2/OpenTraj/SDD/bookstore/video0/reference.jpg'
    # homog_file = ''

    # parser = ParserGC()
    # annot_file = '/home/cyrus/workspace2/OpenTraj/GC/Annotation'
    # media_file = '/home/cyrus/workspace2/OpenTraj/GC/reference.jpg'
    # homog_file = ''

    if os.path.exists(homog_file):
        Hinv = np.linalg.inv(np.loadtxt(homog_file))
    else:
        Hinv = np.eye(3, 3)

    parser.load(annot_file)
    play(parser, Hinv, media_file)