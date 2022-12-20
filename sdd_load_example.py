import os, yaml, sys
import pickle
import pandas as pd

from opentraj.toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir
sys.path.append('.')

scenes = {
    'bookstore' : 7,
    'coupa' : 4,
    'deathCircle' : 5,
    'gates' : 9,
    'hyang' : 15,
    'little' : 4,
    'nexus' : 12,
    'quad' : 4
}
# Fixme: set proper OpenTraj directory
sdd_root = os.path.join('/Users/marcemq/repos/OpenTraj', 'datasets', 'SDD')

sdd_data = {key: pd.DataFrame() for key in scenes.keys() } 

for scene_name, total_videos_per_scene in scenes.items():
    scene_video_ids = ['video'+str(i) for i in range(total_videos_per_scene)]
    traj_datasets_per_scene = []

    for scene_video_id in scene_video_ids:
        annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')

        # load the homography values
        with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
            scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
        scale = scales_yaml_content[scene_name][scene_video_id]['scale']

        itraj_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                        drop_lost_frames=False, use_kalman=False, label='Pedestrian')
        traj_datasets_per_scene.append(pd.concat([itraj_dataset.data.iloc[:, : 4], itraj_dataset.data.iloc[:, 8:9]], axis=1))

    pickle_out = open(scene_name+'.pickle',"wb")
    pickle.dump(pd.concat(traj_datasets_per_scene), pickle_out, protocol=2)
    pickle_out.close()
