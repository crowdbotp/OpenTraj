# Jean-Bernard Hayet
# jbhayet@cimat.mx

import pathlib

import numpy as np
import pandas as pd

from toolkit.core.trajdataset import TrajDataset


def load_ind(path, **kwargs):
    traj_dataset = TrajDataset()
    # Note: we assume here that the path that is passed is the one to the tracks CSV.

    # Read the tracks
    raw_dataset = pd.read_csv(path, sep=",", header=0,
                              names=["recordingId", "trackId", "frame", "trackLifetime", "xCenter", "yCenter",
                                     "heading", "width", "length", "xVelocity", "yVelocity", "xAcceleration",
                                     "yAcceleration", "lonVelocity", "latVelocity", "lonAcceleration",
                                     "latAcceleration"])

    # Read the recording data
    data_path = pathlib.Path(path)
    datadir_path = data_path.parent
    recording_path = str(datadir_path) + '/{:02d}_recordingMeta.csv'.format(raw_dataset['recordingId'][0])
    recording_data = pd.read_csv(recording_path, sep=",", header=0,
                                 names=["recordingId", "locationId", "frameRate", "speedLimit", "weekday", "startTime",
                                        "duration", "numTracks", "numVehicles", "numVRUs", "latLocation", "lonLocation",
                                        "xUtmOrigin", "yUtmOrigin", "orthoPxToMeter"])
    traj_dataset.title = kwargs.get('title', "inD")

    # Read the meta-tracks data
    tracks_path = str(datadir_path) + '/{:02d}_tracksMeta.csv'.format(raw_dataset['recordingId'][0])
    tracks_data = pd.read_csv(tracks_path, sep=",", header=0,
                              names=["recordingId", "trackId", "initialFrame", "finalFrame", "numFrames", "width",
                                     "length", "class"])
    # Get the ids of pedestrians only
    ped_ids = tracks_data[tracks_data["class"] == "pedestrian"]["trackId"].values
    raw_dataset = raw_dataset[raw_dataset['trackId'].isin(ped_ids)]

    # Copy columns
    traj_dataset.data[["frame_id", "agent_id",
                      "pos_x", "pos_y", "vel_x", "vel_y"]] = \
        raw_dataset[["frame", "trackId",
                     "xCenter", "yCenter", "xVelocity", "yVelocity"]]

    traj_dataset.data["label"] = "pedestrian"
    scene_id = kwargs.get("scene_id", recording_data["locationId"][0])
    traj_dataset.data["scene_id"] = scene_id
    # print("location_id = ", recording_data["locationId"][0])

    # post-process
    fps = int(recording_data["frameRate"][0])  # fps = 25
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset
