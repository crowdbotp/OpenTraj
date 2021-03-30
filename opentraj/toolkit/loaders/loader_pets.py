# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import xml.etree.ElementTree as et

import numpy as np
import pandas as pd

from toolkit.core.trajdataset import TrajDataset
from toolkit.utils.calibration.camera_calibration_tsai import *


def load_pets(path, **kwargs):
    """
    :param path: address of annotation file
    :param kwargs:
    :param  calib_path: address of calibration file
    :return: TrajectoryDataset object
    """
    traj_dataset = TrajDataset()

    annot_xtree = et.parse(path)
    annot_xroot = annot_xtree.getroot()  # dataset

    cp, cc = None, None  # calibration parameters

    # load calibration
    calib_path = kwargs.get('calib_path', "")
    if calib_path:
        cp = CameraParameters()
        cc = CalibrationConstants()

        calib_xtree = et.parse(calib_path)
        calib_xroot = calib_xtree.getroot()  # Camera

        geometry_node = calib_xroot.find("Geometry")
        width = int(geometry_node.attrib["width"])
        height = int(geometry_node.attrib["height"])

        cp.Ncx = float(geometry_node.attrib["ncx"])
        cp.Nfx = float(geometry_node.attrib["nfx"])
        cp.dx = float(geometry_node.attrib["dx"])
        cp.dy = float(geometry_node.attrib["dy"])
        cp.dpx = float(geometry_node.attrib["dpx"])
        cp.dpy = float(geometry_node.attrib["dpy"])

        intrinsic_node = calib_xroot.find("Intrinsic")
        cc.f = float(intrinsic_node.attrib["focal"])
        cc.kappa1 = float(intrinsic_node.attrib["kappa1"])  # 1st order radial distortion

        cp.Cx = float(intrinsic_node.attrib["cx"])
        cp.Cy = float(intrinsic_node.attrib["cy"])
        cp.sx = float(intrinsic_node.attrib["sx"])

        extrinsic_node = calib_xroot.find("Extrinsic")
        cc.Tx = float(extrinsic_node.attrib["tx"])
        cc.Ty = float(extrinsic_node.attrib["ty"])
        cc.Tz = float(extrinsic_node.attrib["tz"])
        cc.Rx = float(extrinsic_node.attrib["rx"])
        cc.Ry = float(extrinsic_node.attrib["ry"])
        cc.Rz = float(extrinsic_node.attrib["rz"])

        cc.calc_rr()  # Calculate Rotation Matrix

    loaded_data = []  # frame_id, agent_id, pos_x, pos_y, xc, yc, h, w
    for frame_node in annot_xroot:
        objectlist_node = frame_node.find("objectlist")  # .text
        object_nodes = objectlist_node.findall("object")
        frame_id = int(frame_node.attrib.get("number"))

        for obj_node in object_nodes:
            agent_id = obj_node.attrib["id"]

            box_node = obj_node.find("box")
            xc = float(box_node.attrib["xc"])
            yc = float(box_node.attrib["yc"])
            h = float(box_node.attrib["h"])
            w = float(box_node.attrib["w"])

            x_ground = xc
            y_ground = yc + h/2

            if cp:
                pos_x, pos_y = image_coord_to_world_coord(x_ground, y_ground, 0, cp, cc)
            else:
                pos_x, pos_y = np.nan, np.nan

            loaded_data.append([frame_id, agent_id, pos_x / 1000., pos_y / 1000., xc, yc, h, w])

    data_columns = ["frame_id", "agent_id", "pos_x", "pos_y",
                    "xc", "yc", "h", "w"]
    raw_dataset = pd.DataFrame(np.array(loaded_data), columns=data_columns)

    traj_dataset.title = kwargs.get('title', "PETS")

    # copy columns
    traj_dataset.data[["frame_id", "agent_id",
                       "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id",
                     "pos_x", "pos_y"]]
    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 7)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset
