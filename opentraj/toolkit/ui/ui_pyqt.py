import numpy as np
import cv2
import os
import argparse
import time

# from parser.parser_eth import ParserETH
# from parser.parser_sdd import ParserSDD
# from parser.parser_gc  import ParserGC
# from parser.parser_hermes import ParserHermes

from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir
from toolkit.loaders.loader_gcs import load_gcs
from toolkit.loaders.loader_hermes import load_bottleneck

from toolkit.ui.pyqt.qtui.opentrajui import OpenTrajUI


def error_msg(msg):
    print('Error:', msg)
    exit(-1)


class Play:
    def __init__(self, gui_mode):
        bg_im = None
        self.gui_mode_pyqt = (args.gui_mode == 'pyqt')
        self.gui_mode_opencv = (args.gui_mode == 'opencv')
        if self.gui_mode_pyqt:
            self.qtui = OpenTrajUI(reserve_n_agents=100)
            self.agent_index = -1

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
        if self.gui_mode_pyqt:
            self.qtui.update_im(im)
            self.qtui.erase_paths()
            self.qtui.erase_circles()
        if timestamp >= 0:
            if self.gui_mode_pyqt:
                self.qtui.setTimestamp(timestamp)
            elif self.gui_mode_opencv:
                cv2.putText(self.bg_im, '%d' % timestamp, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    def draw_trajectory(self, id, ll, color, width):
        if self.gui_mode_pyqt:
            self.qtui.draw_path(ll[..., ::-1], color, [width])

        elif self.gui_mode_opencv:
            for tt in range(ll.shape[0] - 1):
                cv2.line(self.bg_im, (ll[tt][1], ll[tt][0]), (ll[tt + 1][1], ll[tt + 1][0]), color, width)

    def draw_agent(self, id, pos, radius, color, width):
        if self.gui_mode_pyqt:
            self.qtui.draw_circle(pos, radius, color, width)
        elif self.gui_mode_opencv:
            cv2.circle(self.bg_im, (pos[1], pos[0]), radius, color, width)

    def play(self, traj_dataset, Hinv, media_file):
        timestamps = sorted(traj_dataset.__t_p_dict__.keys())

        if os.path.exists(media_file):
            if self.is_a_video(media_file):
                cap = cv2.VideoCapture(media_file)
            else:
                ref_im = cv2.imread(media_file)

        ids_t = []
        # for t in range(timestamps[0], timestamps[-1]):
        # for t in range(0, timestamps[-1]):
        t = 0
        pause = False
        while True:
            if self.is_a_video(media_file) and not pause:
                ret, ref_im = cap.read()

            ref_im_copy = np.copy(ref_im)
            self.set_background_im(ref_im, t)
            if t in timestamps:
                xys_t = traj_dataset.__t_p_dict__[t]
                ids_t = traj_dataset.__t_id_dict__[t]

            for i, id in enumerate(ids_t):
                xy_i = np.array(xys_t[i])
                UV_i = self.to_image_frame(Hinv, xy_i)

                # fetch entire trajectory
                traj_i = traj_dataset.__id_p_dict__[id]
                TRAJ_i = self.to_image_frame(Hinv, traj_i)

                self.draw_trajectory(id, TRAJ_i, (255, 255, 0), 2)
                self.draw_agent(id, (UV_i[1], UV_i[0]), 5, (0, 0, 255), 2)

            # if not ids_t:
            #     print('No agent')

            if not pause and t < timestamps[-1]:
                t += 1

            delay_ms = 20
            if self.gui_mode_pyqt:
                self.qtui.processEvents()
                pause = self.qtui.pause
                time.sleep(delay_ms/1000.)
            else:
                # cv2.namedWindow('OpenTraj (Press ESC for exit)', cv2.WINDOW_NORMAL)
                # cv2.imshow('OpenTraj (Press ESC for exit)', ref_im_copy)
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
    argparser.add_argument('--data-root', '--data-root',
                           help='the root address of OpenTraj directory')
    argparser.add_argument('--gui-mode', '--g',
                           default='pyqt', choices=['pyqt', 'opencv'], #  'tkinter' ?
                           help='pick a specific mode for gui'
                                '(default: "pyqt")')
    argparser.add_argument('--dataset', '--dataset',
                           default='eth',
                           choices=['eth',
                                    'hotel',
                                    'zara01',
                                    'zara02',
                                    'students03',
                                    'gc',
                                    'sdd-bookstore0',
                                    'Hermes-xxx'],
                           help='select dataset'
                                '(default: "eth")')
    argparser.add_argument('--background', '--b',
                           default='image', choices=['image', 'video'],
                           help='select background type. video does not exist for all datasets,'
                                'you might need to download it first.'
                                '(default: "image")')

    args = argparser.parse_args()
    opentraj_root = args.data_root
    traj_dataset = None

    # #============================ ETH =================================
    if args.dataset == 'eth':
        annot_file = os.path.join(opentraj_root, 'ETH/seq_eth/obsmat.txt')
        traj_dataset = load_eth(annot_file)
        homog_file = os.path.join(opentraj_root, 'ETH/seq_eth/H.txt')
        if args.background == 'image':
            media_file = os.path.join(opentraj_root, 'ETH/seq_eth/reference.png')
        elif args.background == 'video':
            media_file = os.path.join(opentraj_root, 'ETH/seq_eth/video.avi')
        else:
            error_msg('background type is invalid')

    elif args.dataset == 'hotel':
        annot_file = os.path.join(opentraj_root, 'ETH/seq_hotel/obsmat.txt')
        traj_dataset = load_eth(annot_file)
        homog_file = os.path.join(opentraj_root, 'ETH/seq_hotel/H.txt')
        # media_file = os.path.join(opentraj_root, 'ETH/seq_hotel/reference.png')
        media_file = os.path.join(opentraj_root, 'ETH/seq_hotel/video.avi')

    # #============================ UCY =================================
    # elif args.dataset == 'zara01':
    #     traj_dataset = ParserETH()
    #     parser = ParserETH()
    #     annot_file = os.path.join(opentraj_root, 'UCY/zara01/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/zara01/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/zara01/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/zara01/video.avi')
    #
    # elif args.dataset == 'zara01':
    #     traj_dataset = ParserETH()
    #     annot_file = os.path.join(opentraj_root, 'UCY/zara02/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/zara02/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/zara02/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/zara02/video.avi')
    #
    # elif args.dataset == 'students03':
    #     traj_dataset = ParserETH()
    #     # annot_file = os.path.join(opentraj_root, 'UCY/st3_dataset/obsmat_px.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/reference.png')
    #     # homog_file = os.path.join(opentraj_root, 'UCY/st3_dataset/H_iw.txt')
    #     # # homog_file = ''
    #
    #     annot_file = os.path.join(opentraj_root, 'UCY/st3_dataset/obsmat.txt')
    #     homog_file = os.path.join(opentraj_root, 'UCY/st3_dataset/H.txt')
    #     # media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/reference.png')
    #     media_file = os.path.join(opentraj_root, 'UCY/st3_dataset/video.avi')

    # #============================ SDD =================================
    # dataset_parser = ParserSDD()
    # annot_file = os.path.join(opentraj_root, 'SDD/bookstore/video0/annotations.txt')
    # media_file = os.path.join(opentraj_root, 'SDD/bookstore/video0/reference.jpg')
    # homog_file = ''

    # #============================ GC ==================================
    # elif args.dataset == 'gc':
    #     gc_world_coord = True
    #     traj_dataset = ParserGC(world_coord=gc_world_coord)
    #     annot_file = os.path.join(opentraj_root, 'GC/Annotation')  # image coordinate
    #     if gc_world_coord:
    #         homog_file = os.path.join(opentraj_root, 'GC/H-world.txt')
    #         media_file = os.path.join(opentraj_root, 'GC/plan.png')
    #     else:
    #         homog_file = os.path.join(opentraj_root, 'GC/H-image.txt')
    #         media_file = os.path.join(opentraj_root, 'GC/reference.jpg')

    # ========================== HERMES =================================
    # dataset_parser = ParserHermes()
    # annot_file = os.path.join(opentraj_root, 'HERMES/Corridor-1D/uo-070-180-180.txt')
    # annot_file = os.path.join(opentraj_root, 'HERMES/Corridor-2D/boa-300-050-070.txt')
    # media_file = os.path.join(opentraj_root, 'HERMES/cor-180.jpg')
    # homog_file = os.path.join(opentraj_root, 'HERMES/H.txt')

    if not traj_dataset:
        error_msg('dataset name is invalid')

    Homog = (np.loadtxt(homog_file)) if os.path.exists(homog_file) else np.eye(3)
    Hinv = np.linalg.inv(Homog)

    play = Play(args.gui_mode)
    play.play(traj_dataset, Hinv, media_file)
    # qtui.app.exe()
