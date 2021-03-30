# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import pandas as pd
import numpy as np
from toolkit.core.trajdataset import TrajDataset
import datetime


def load_cff(path, **kwargs):
    traj_dataset = TrajDataset()

    # read from csv => python str
    # sample line:  2012-09-18T06:25:00:036;PIE;17144;50515;1
    # columns = ["Year", "Month", "Day", "Hour", "min", "sec", "msec", "place", "x_mm", "y_mm", "agent_id"]
    with open(path, 'r') as inp_file:
        file_content = inp_file.read()
        file_content = file_content.replace('T', '-').replace(':', '-').replace(';', '-').replace('\n', '-')
    segments = file_content.split('-')
    segments.remove('')
    year = np.array(segments[0::11], dtype=int)
    month = np.array(segments[1::11], dtype=int)
    day = np.array(segments[2::11], dtype=int)
    hour = np.array(segments[3::11], dtype=int)
    minute = np.array(segments[4::11], dtype=int)
    second = np.array(segments[5::11], dtype=int)
    milli_sec = np.array(segments[6::11], dtype=int)
    place = np.array(segments[7::11], dtype=str)
    x_mm = np.array(segments[8::11], dtype=float)
    y_mm = np.array(segments[9::11], dtype=float)
    agent_id = np.array(segments[10::11], dtype=int)
    # skip year and month
    timestamp = ((day * 24 + hour) * 60 + minute) * 60 + second + milli_sec/1000.
    fps = 10

    traj_dataset.title = kwargs.get('title', "Train Terminal")

    raw_dataset = pd.DataFrame({"timestamp": timestamp,
                                "frame_id": (timestamp * fps).astype(int),
                                "agent_id": agent_id,
                                "pos_x": x_mm / 1000.,
                                "pos_y": y_mm / 1000.,
                                })

    # raw_dataset["scene_id"] = place
    scene_id = kwargs.get('scene_id', 0)
    raw_dataset["scene_id"] = scene_id

    # copy columns
    traj_dataset.data[["scene_id", "timestamp", "frame_id", "agent_id",
                       "pos_x", "pos_y"]] = \
        raw_dataset[["scene_id", "timestamp", "frame_id", "agent_id",
                     "pos_x", "pos_y"]]

    traj_dataset.data["label"] = "pedestrian"

    # post-process
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', True)  # use kalman smoother by default
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset


# test
if __name__ == '__main__':
    dataset = load_cff("/home/cyrus/workspace2/OpenTraj/datasets/CFF/al_position2013-02-10.csv")
    print(dataset.get_agent_ids())
