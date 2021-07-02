import json

import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from toolkit.loaders.loader_metafile import load_metafile
from toolkit.ui.ui_projectpoint import to_image_frame
import matplotlib
matplotlib.use('TkAgg')


def error_msg(msg):
    print('Error:', msg)
    exit(-1)


class Play:
    def __init__(self):
        pass

    def is_a_video(self, filename):
        return '.mp4' in filename or '.avi' in filename

    def to_image_frame(self, Hinv, loc):
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

    def set_background_im(self, im, timestamp=-1):
        self.bg_im = im.copy()
        cv2.putText(self.bg_im, '%d' % timestamp, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def draw_trajectory(self, id, ll, color, width):
        for tt in range(ll.shape[0] - 1):
            cv2.line(self.bg_im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), color, width)

    def draw_agent(self, id, pos, radius, color, width):
        cv2.circle(self.bg_im, (pos[1], pos[0]), radius, color, width)

    def play(self, traj_dataset, Hinv, media_file):
        frame_ids = sorted(traj_dataset.data['frame_id'].unique())

        if os.path.exists(media_file):
            if self.is_a_video(media_file):
                cap = cv2.VideoCapture(media_file)
            else:
                ref_im = cv2.imread(media_file)

        ids_t = []
        # for frame_id in range(frame_ids[0], frame_ids[-1]):
        # for frame_id in range(0, frame_ids[-1]):
        frame_id = 0
        pause = False
        while True:
            if self.is_a_video(media_file) and not pause:
                ret, ref_im = cap.read()

            # ref_im_copy = np.copy(ref_im)
            self.set_background_im(ref_im, frame_id)
            if frame_id in frame_ids:
                xys_t = traj_dataset.data[['pos_x', 'pos_y']].loc[traj_dataset.data["frame_id"] == frame_id].to_numpy()
                ids_t = traj_dataset.data['agent_id'].loc[traj_dataset.data["frame_id"] == frame_id].to_numpy()

            all_trajs = traj_dataset.data[
                                          (traj_dataset.data['agent_id'].isin(ids_t)) &
                                          (traj_dataset.data['frame_id'] <= frame_id) &
                                          (traj_dataset.data['frame_id'] > frame_id - 50)  # Todo: replace it with timestamp
                                          ].groupby('agent_id')
            ids_t = [key for key, value in all_trajs]
            all_trajs = [value[['pos_x', 'pos_y']].to_numpy() for key, value in all_trajs]

            for i, id in enumerate(ids_t):
                xy_i = np.array(xys_t[i])
                UV_i = self.to_image_frame(Hinv, xy_i)

                # fetch entire trajectory

                traj_i = all_trajs[i]
                TRAJ_i = self.to_image_frame(Hinv, traj_i)

                self.draw_trajectory(id, TRAJ_i, (255, 255, 0), 2)
                self.draw_agent(id, (UV_i[0], UV_i[1]), 5, (0, 0, 255), 2)

            # if not ids_t:
            #     print('No agent')

            if not pause and frame_id < frame_ids[-1]:
                frame_id += 1

            delay_ms = 20
            cv2.namedWindow('OpenTraj (Press ESC for exit)', cv2.WINDOW_NORMAL)
            cv2.imshow('OpenTraj (Press ESC for exit)', self.bg_im)
            key = cv2.waitKey(delay_ms * (1 - pause)) & 0xFF
            if key == 27:  # press ESCAPE to quit
                break
            elif key == 32:  # press SPACE to pause/play
                pause = not pause
# def play(parser, Hinv, media_file):
#     timestamps = sorted(parser.t_p_dict.keys())
#
#     if os.path.exists(media_file):
#         if is_a_video(media_file):
#             cap = cv2.VideoCapture(media_file)
#         else:
#             ref_im = cv2.imread(media_file)
#
#     pause = False
#     ids_t = []
#     # for t in range(timestamps[0], timestamps[-1]):
#     for t in range(0, timestamps[-1]):
#         if is_a_video(media_file):
#             ret, ref_im = cap.read()
#
#         ref_im_copy = np.copy(ref_im)
#         if t in timestamps:
#             xys_t = parser.t_p_dict[t]
#             ids_t = parser.t_id_dict[t]
#
#         for i, id in enumerate(ids_t):
#             xy_i = np.array(xys_t[i])
#             UV_i = to_image_frame(Hinv, xy_i)
#
#             # fetch entire trajectory
#             traj_i = parser.id_p_dict[id]
#             TRAJ_i = to_image_frame(Hinv, traj_i)
#             draw_line(ref_im_copy, TRAJ_i, (255, 255, 0), 2)
#             cv2.circle(ref_im_copy, (UV_i[1], UV_i[0]), 5, (0, 0, 255), 2)
#
#         cv2.putText(ref_im_copy, '%d' % t, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
#         cv2.namedWindow('OpenTraj (Press ESC for exit)', cv2.WINDOW_NORMAL)
#         cv2.imshow('OpenTraj (Press ESC for exit)', ref_im_copy)
#         delay_ms = 100
#         key = cv2.waitKey(delay_ms * (1-pause)) & 0xFF
#         if key == 27:     # press ESCAPE to quit
#             break
#         elif key == 32:   # press SPACE to pause/play
#             pause = not pause


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='OpenTraj - Human Trajectory Dataset Package')
    argparser.add_argument('--opentraj-root', '--opentraj-root',
                           help='the root address of OpenTraj directory')

    argparser.add_argument('--metafile', '--metafile',
                           help='dataset meta file (*.json)')

    argparser.add_argument('--background', '--b',
                           default='image', choices=['image', 'video'],
                           help='select background type. video does not exist for all datasets,'
                                'you might need to download it first.'
                                '(default: "image")')

    args = argparser.parse_args()
    opentraj_root = args.opentraj_root
    traj_dataset = None

    # #============================ ETH & ZARA =================================
    if not args.metafile:
        error_msg('Please Enter a valid dataset metafile (*.json)')

    with open(args.metafile) as json_file:
        data = json.load(json_file)
    if 'calib_to_world_path' in data:
        homog_to_world_file = os.path.join(opentraj_root, data['calib_to_world_path'])
        Homog_to_world = (np.loadtxt(homog_to_world_file)) if os.path.exists(homog_to_world_file) else np.eye(3)
    else:
        homog_to_world_file = ""

    if 'calib_to_camera_path' in data:
        homog_to_camera_file = os.path.join(opentraj_root, data['calib_to_camera_path'])
        Homog_to_camera = (np.loadtxt(homog_to_camera_file)) if os.path.exists(homog_to_camera_file) else np.eye(3)
        Hinv = np.linalg.inv(Homog_to_camera)
    else:
        homog_to_camera_file = ""

    traj_dataset = load_metafile(opentraj_root, args.metafile, homog_file=homog_to_world_file)


    if args.background == 'image':
        media_file = os.path.join(opentraj_root, data['ref_image'])
    elif args.background == 'video':
        media_file = os.path.join(opentraj_root, data['video'])
    else:
        error_msg('background type is invalid')

    if not traj_dataset:
        error_msg('dataset name is invalid')

    play = Play()
    play.play(traj_dataset, Hinv, media_file)
