# Francisco Valente Castro
# francisco.valente@cimat.mx

import cv2
import pathlib
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation
from toolkit.core.trajdataset import TrajDataset


def read_projection_parameters(path):
    # Read calibration file
    calibration_file = open(path, 'r')

    # Save calibration parameters in dictionary
    param = {}
    for line in calibration_file:
        name, _, value = line.split()
        param[name] = float(value)

    # Rotation vector
    rvec = np.array([param['RotationX'],
                     param['RotationY'],
                     param['RotationZ'],
                     param['RotationW']])

    # Translation vector
    tvec = np.array([param['TranslationX'],
                     param['TranslationY'],
                     param['TranslationZ']])

    # Camera matrix
    fx = param['FocalLengthX']
    fy = param['FocalLengthY']
    px = param['PrincipalPointX']
    py = param['PrincipalPointY']
    s = param['Skew']

    cameraMatrix = np.array([[fx, s, px],
                             [0.0, fy, py],
                             [0.0, 0.0, 1.0]])

    # Distortion coefficients
    distCoeffs = np.array([param['DistortionK1'],
                           param['DistortionK2'],
                           param['DistortionP1'],
                           param['DistortionP2']])

    return rvec, tvec, cameraMatrix, distCoeffs


def obtainObjectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs):
    # Obtain variables values
    dim_0 = pts.shape[0]

    # Add new dimension to use undistort points
    pts = pts.reshape([pts.shape[0], 1, 2])

    # Undistort points and add homogeneuos coordinate
    undPts = cv2.undistortPoints(pts, cameraMatrix, distCoeffs)
    undPts = undPts.reshape([undPts.shape[0], 2])
    undPts = np.concatenate([undPts, np.ones([dim_0, 1])], axis=1)

    # Construct inverse rotation matrix from quaternion
    qx, qy, qz, qw = rvec
    R = Rotation.from_quat([qx, qy, qz, qw]).inv()
    rotation = R.as_matrix()

    # Translation vector and normal vector
    normal = np.array([[0], [0], [1]])
    tvec = np.array(tvec).reshape([3, 1])

    # Rotate points
    rotatedPts = (rotation @ undPts.T).T
    num = -normal.T @ tvec
    denom = (normal.T @ rotatedPts.T).T

    # Get un-projected points
    unprojPts = num / denom * rotatedPts + tvec.T

    return unprojPts


def load_town_center(path, **kwargs):
    # Construct dataset
    traj_dataset = TrajDataset()

    # Note: we assume here that the path that is passed is the one to the tracks CSV.
    # Read the tracks
    raw_dataset = pd.read_csv(path, sep=",", header=0,
                              names=["personNumber", "frameNumber", "headValid", "bodyValid", "headLeft", "headTop",
                                     "headRight", "headBottom", "bodyLeft", "bodyTop", "bodyRight", "bodyBottom"])

    # Get bottom (feet) of bounding boxes
    raw_dataset["body_x"] = (raw_dataset["bodyLeft"] + raw_dataset["bodyRight"]) / 2.0
    raw_dataset["body_y"] = raw_dataset["bodyBottom"]

    raw_dataset["head_x"] = (raw_dataset["headLeft"] + raw_dataset["headRight"]) / 2.0
    raw_dataset["head_y"] = (raw_dataset["headTop"] + raw_dataset["headBottom"]) / 2.0

    # Required information
    raw_dataset["label"] = "pedestrian"

    # Read camera calibration
    calibration_path = kwargs.get('calib_path', 'none')
    rvec, tvec, cameraMatrix, distCoeffs =\
        read_projection_parameters(calibration_path)

    # Obtain real world coordinates from image
    pts = np.array([raw_dataset["body_x"], raw_dataset["body_y"]]).T
    objPts = obtainObjectPoints(pts, rvec, tvec,
                                cameraMatrix, distCoeffs)

    # Add object points to raw dataset
    raw_dataset['pos_x'] = objPts[:, 0]
    raw_dataset['pos_y'] = objPts[:, 1]
    raw_dataset['pos_z'] = objPts[:, 2]

    # Remove invalid body bounding boxes
    raw_dataset = raw_dataset[raw_dataset.bodyValid == 1]

    # Copy columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frameNumber", "personNumber", "pos_x", "pos_y"]]

    # FixMe: for debug
    traj_dataset.data[["body_x", "body_y"]] = \
        raw_dataset[["body_x", "body_y"]].astype(int)

    # Recording information
    traj_dataset.title = kwargs.get('title', "Town-Center")
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 25)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


if __name__ == "__main__":
    import sys, os, cv2
    towncenter_root = sys.argv[1]

    traj_ds = load_town_center(towncenter_root + '/TownCentre-groundtruth-top.txt',
                               calib_path=towncenter_root + '/TownCentre-calibration-ci.txt',
                               # use_kalman=True,
                               sampling_rate=10
                               )

    video_address = os.path.join(towncenter_root, 'TownCentreXVID.avi')
    cap = cv2.VideoCapture(video_address)

    traj_groups = traj_ds.get_trajectories()
    trajs = [g for _, g in traj_groups]

    for ii, traj in enumerate(trajs):
        dur = traj["timestamp"].iloc[-1] - traj["timestamp"].iloc[0]
        print("duration of traj %d is %.2f(s)" % (ii, dur))

    frame_id = -1
    while True:
        frame_id += 1
        ret, im = cap.read()

        frame_locs = traj_ds.get_frames(frame_ids=[frame_id])
        if len(frame_locs):
            agnet_i_t = frame_locs[0][["agent_id", "body_x", "body_y"]].to_numpy()
            for ii, loc_i in enumerate(agnet_i_t[:, 1:]):
                agent_id = agnet_i_t[ii, 0]
                cv2.circle(im, (loc_i[0], loc_i[1]), 5, (0, 0, 200), 2)
                cv2.putText(im, "%d" % agent_id, (loc_i[0], loc_i[1]),
                            0, 2, 255)

        cv2.imshow('im', im)
        k = cv2.waitKey(10)
        if k == 27:
            break
