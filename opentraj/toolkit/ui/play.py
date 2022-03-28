import numpy as np
import cv2
import os

from parser.parser_eth import ParserETH
from parser.parser_sdd import ParserSDD
from parser.parser_gc  import ParserGC
from parser.parser_hermes import ParserHermes


def is_a_video(filename):
    return '.mp4' in filename or '.avi' in filename


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
    timestamps = sorted(parser.t_p_dict.keys())

    if os.path.exists(media_file):
        if is_a_video(media_file):
            cap = cv2.VideoCapture(media_file)
        else:
            ref_im = cv2.imread(media_file)

    pause = False
    ids_t = []
    # for t in range(timestamps[0], timestamps[-1]):
    for t in range(0, timestamps[-1]):
        if is_a_video(media_file):
            ret, ref_im = cap.read()

        ref_im_copy = np.copy(ref_im)
        if t in timestamps:
            xys_t = parser.t_p_dict[t]
            ids_t = parser.t_id_dict[t]

        for i, id in enumerate(ids_t):
            xy_i = np.array(xys_t[i])
            UV_i = to_image_frame(Hinv, xy_i)

            # fetch entire trajectory
            traj_i = parser.id_p_dict[id]
            TRAJ_i = to_image_frame(Hinv, traj_i)
            line_cv(ref_im_copy, TRAJ_i, (255, 255, 0), 2)
            cv2.circle(ref_im_copy, (UV_i[1], UV_i[0]), 5, (0, 0, 255), 2)

        cv2.putText(ref_im_copy, '%d' % t, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.imshow('OpenTraj (Press ESC for exit)', ref_im_copy)
        delay_ms = 100
        key = cv2.waitKey(delay_ms * (1-pause)) & 0xFF
        if key == 27:     # press ESCAPE to quit
            break
        elif key == 32:   # press SPACE to pause/play
            pause = not pause


if __name__ == '__main__':
    opentraj_path = '/home/cyrus/workspace2/OpenTraj'  # FIXME
    # #============================ ETH =================================
    # parser = ParserETH()

    # annot_file = os.path.join(opentraj_path, 'ETH/seq_eth/obsmat.txt')
    # homog_file = os.path.join(opentraj_path, 'ETH/seq_eth/H.txt')
    # # media_file = os.path.join(opentraj_path, 'ETH/seq_eth/reference.png')
    # media_file = os.path.join(opentraj_path, 'ETH/seq_eth/video.avi')

    # annot_file = os.path.join(opentraj_path, 'ETH/seq_hotel/obsmat.txt')
    # homog_file = os.path.join(opentraj_path, 'ETH/seq_hotel/H.txt')
    # # media_file = os.path.join(opentraj_path, 'ETH/seq_hotel/reference.png')
    # media_file = os.path.join(opentraj_path, 'ETH/seq_hotel/video.avi')

    # #============================ UCY =================================
    parser = ParserETH()

    # annot_file = os.path.join(opentraj_path, 'UCY/data_zara01/obsmat.txt')
    # homog_file = os.path.join(opentraj_path, 'UCY/data_zara01/H.txt')
    # # media_file = os.path.join(opentraj_path, 'UCY/data_zara01/reference.png')
    # media_file = os.path.join(opentraj_path, 'UCY/data_zara01/video.avi')

    # annot_file = os.path.join(opentraj_path, 'UCY/data_zara02/obsmat.txt')
    # homog_file = os.path.join(opentraj_path, 'UCY/data_zara02/H.txt')
    # media_file = os.path.join(opentraj_path, 'UCY/data_zara02/reference.png')
    # media_file = os.path.join(opentraj_path, 'UCY/data_zara02/video.avi')

    # annot_file = os.path.join(opentraj_path, 'UCY/st3_dataset/obsmat_px.txt')
    # media_file = os.path.join(opentraj_path, 'UCY/st3_dataset/reference.png')
    # homog_file = os.path.join(opentraj_path, 'UCY/st3_dataset/H_iw.txt')
    # # homog_file = ''

    annot_file = os.path.join(opentraj_path, 'UCY/st3_dataset/obsmat.txt')
    homog_file = os.path.join(opentraj_path, 'UCY/st3_dataset/H.txt')
    # media_file = os.path.join(opentraj_path, 'UCY/st3_dataset/reference.png')
    media_file = os.path.join(opentraj_path, 'UCY/st3_dataset/video.avi')

    # #============================ SDD =================================
    # parser = ParserSDD()
    # annot_file = os.path.join(opentraj_path, 'SDD/bookstore/video0/annotations.txt')
    # media_file = os.path.join(opentraj_path, 'SDD/bookstore/video0/reference.jpg')
    # homog_file = ''

    # #============================ GC ==================================
    # parser = ParserGC()
    # annot_file = os.path.join(opentraj_path, 'GC/Annotation')
    # media_file = os.path.join(opentraj_path, 'GC/reference.jpg')
    # homog_file = ''

    # ========================== HERMES =================================
    # parser = ParserHermes()
    # annot_file = os.path.join(opentraj_path, 'HERMES/Corridor-1D/uo-070-180-180/uo-070-180-180_combined_MB.txt')
    # media_file = os.path.join(opentraj_path, 'HERMES/cor-180.jpg')
    # homog_file = os.path.join(opentraj_path, 'HERMES/H.txt')

    parser.load(annot_file)
    n_peds = len(parser.id_p_dict.keys())
    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    Hinv = np.linalg.inv(Homog)
    play(parser, Hinv, media_file)
