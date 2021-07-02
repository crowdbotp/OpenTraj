# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from toolkit.ui.ui_projectpoint import to_image_frame
from toolkit.ui.ui_constants import RED_COLOR


def _visualize_trajlet_on_frame(trajlet, bg_im, homog, color_rgb=RED_COLOR, width=2):
    tl = to_image_frame(homog, trajlet[:, :2])
    for tt in range(tl.shape[0] - 1):
        cv2.line(bg_im, (tl[tt][1], tl[tt][0]), (tl[tt + 1][1], tl[tt + 1][0]),
                 color_rgb[::-1], width)

    # cv2.imshow("vis", bg_im)
    # cv2.waitKey(2000)
    return bg_im


def retrieve_bg_image(ds_name, opentraj_root, timestamp=-1):
    homog = None
    bg_frame = None

    if ds_name == 'ETH-Univ':
        # bg_frame_file = os.path.join(opentraj_root, "datasets/ETH/seq_eth/bg.png")
        # bg_frame = cv2.imread(bg_frame_file)
        video_file = os.path.join(opentraj_root, "datasets/ETH/seq_eth/video.avi")
        cap = cv2.VideoCapture(video_file)
        frame_id = int(timestamp * 15)
        print()
        print(frame_id)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        _, bg_frame = cap.read()

        homog_file = os.path.join(opentraj_root, "datasets/ETH/seq_eth/H.txt")
        homog = np.linalg.inv(np.loadtxt(homog_file))


    elif ds_name == 'ETH-Hotel':
        # bg_frame_file = os.path.join(opentraj_root, "datasets/ETH/seq_hotel/bg.png")
        # bg_frame = cv2.imread(bg_frame_file)
        video_file = os.path.join(opentraj_root, "datasets/ETH/seq_hotel/video.avi")
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * 25))
        _, bg_frame = cap.read()

        homog_file = os.path.join(opentraj_root, "datasets/ETH/seq_hotel/H.txt")
        homog = np.linalg.inv(np.loadtxt(homog_file))

    elif ds_name == 'UCY-Zara':
        # bg_frame_file = os.path.join(opentraj_root, "datasets/UCY/zara01/bg.png")
        # bg_frame = cv2.imread(bg_frame_file)
        video_file = os.path.join(opentraj_root, "datasets/UCY/zara01/video.avi")
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * 25))
        _, bg_frame = cap.read()


        homog_file = os.path.join(opentraj_root, "datasets/UCY/zara01/H-cam.txt")
        homog = np.linalg.inv(np.loadtxt(homog_file))

        dummy = 1

    elif ds_name == 'UCY-Univ':
        return None, None

    # elif ds_name == 'PETS-S2l1':

    elif ds_name == 'SDD-coupa':
        return None, None

    elif ds_name == 'SDD-bookstore':
        return None, None

    elif ds_name == 'SDD-deathCircle':
        return None, None

    elif ds_name == 'SDD-gates':
        return None, None

    elif ds_name == 'SDD-hyang':
        return None, None

    elif ds_name == 'SDD-little':
        return None, None

    elif ds_name == 'SDD-nexus':
        return None, None

    elif ds_name == 'SDD-quad':
        return None, None

    elif ds_name == 'GC':
        return None, None

    elif ds_name == 'InD-1':  # location_id = 1
        return None, None

    elif ds_name == 'InD-2':  # location_id = 2
        return None, None

    elif ds_name == 'InD-3':  # location_id = 3
        return None, None

    elif ds_name == 'InD-4':  # location_id = 4
        return None, None

    elif ds_name == 'KITTI':
        return None, None

    elif ds_name == 'LCas-Minerva':
        return None, None

    elif ds_name == 'WildTrack':
        return None, None

    elif ds_name == 'Edinburgh':  # 'Edinburgh-01Jul', 'Edinburgh-01Aug', 'Edinburgh-01Sep'
        return None, None

    # Bottleneck (Hermes)
    elif ds_name == 'BN-1d-w180':
        # bg_frame_file = os.path.join(opentraj_root, "datasets/HERMES/figs_and_plots/cor-180.jpg")
        # bg_frame = cv2.imread(bg_frame_file)
        # homog = np.eye(3)
        # homog[0, 0], homog[1, 1] = 0.01, 0.01
        return None, None

    elif ds_name == 'BN-2d-w160':
        # bg_frame_file = os.path.join(opentraj_root, "datasets/HERMES/figs_and_plots/cor-180.jpg")
        # bg_frame = cv2.imread(bg_frame_file)

        video_file = "/media/cyrus/workspace/Seyfried/bo-videos/bo-360-160-160_cam1_iv5.wmv"
        cap = cv2.VideoCapture(video_file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * 16))
        _, bg_frame = cap.read()

        homog = np.array([[100, 0, 0],
                          [0, 100, 0],
                          [0, 0, 1]])


        # homog = np.array([[100, 0, 240],
        #                   [0, 100, 720],
        #                   [0, 0, 1]])

    elif ds_name == 'TownCenter':
        return None, None

    return bg_frame, homog


def visualize_frame(frame, ds_name, opentraj_root):
    bg_frame, homog = retrieve_bg_image(ds_name, opentraj_root, frame[0, 4])

    if homog is None:
        homog = np.eye(3)
    fr = to_image_frame(homog, frame[:, :2])
    # return _visualize_trajlet_on_frame(trajlet, bg_frame, homog)


def visualize_trajlet(trajlet, color_rgb, ds_name, opentraj_root):
    bg_frame, homog = retrieve_bg_image(ds_name, opentraj_root, trajlet[0, 4])

    if homog is None:
        homog = np.eye(3)
    return _visualize_trajlet_on_frame(trajlet, bg_frame, homog, color_rgb)


# test
if __name__ == "__main__":
    pass