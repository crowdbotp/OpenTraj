# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from toolkit.core.trajdataset import TrajDataset
import json
import pandas as pd


def load_forking_path(path, **kwargs):
    traj_dataset = TrajDataset()
    raw_data = []
    with open(path, 'r') as json_file:
        json_content = json_file.read()
        annots_list = json.loads(json_content)
        for annot_dict in annots_list:
            scene_name = annot_dict['scenename']
            x_agents = annot_dict['x_agents']
            ped_controls = annot_dict['ped_controls']

            print(scene_name)
            for frame_id, control_data in ped_controls.items():
                for person_id, _, xyz, direction_vector, speed, time_elasped, is_static in sorted(control_data):
                    print(frame_id, person_id, xyz[:2])
                    raw_data.append([int(frame_id), int(person_id), *xyz[:2]])
                    dummy = 1
            break
    raw_dataset = pd.DataFrame(raw_data, columns = ["frame_id", "agent_id", "pos_x", "pos_y"])
    raw_dataset = raw_dataset.sort_values(by='frame_id', ascending=True).reset_index()
    agents = raw_dataset.groupby("agent_id").apply(list)
    agents = [g for gname, g in raw_dataset.groupby("agent_id")]
    return traj_dataset


if __name__ == '__main__':
    import sys, os
    opentraj_root = sys.argv[1]
    load_forking_path(os.path.join(opentraj_root,
                                 'datasets/Forking-Paths-Garden/forkingpaths_moments_v1/ethucy_v1.json'))

