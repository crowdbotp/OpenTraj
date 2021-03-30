# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import glob
import yaml
import numpy as np
import pandas as pd

from toolkit.core.trajdataset import TrajDataset, merge_datasets
from toolkit.loaders.loader_edinburgh import load_edinburgh
from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_crowds import load_crowds
from toolkit.loaders.loader_gcs import load_gcs
from toolkit.loaders.loader_hermes import load_bottleneck
from toolkit.loaders.loader_ind import load_ind
from toolkit.loaders.loader_kitti import load_kitti
from toolkit.loaders.loader_lcas import load_lcas
from toolkit.loaders.loader_pets import load_pets
from toolkit.loaders.loader_town import load_town_center
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir
from toolkit.loaders.loader_wildtrack import load_wildtrack
from toolkit.loaders.loader_trajnet import load_trajnet
from toolkit.core.trajlet import split_trajectories
from toolkit.baselines.constvel import const_vel

all_dataset_names = [
    'ETH-Univ',
    'ETH-Hotel',

    'UCY-Zara',
    # 'UCY-Zara1',
    # 'UCY-Zara2',

    'UCY-Univ',
    # 'UCY-Univ3',

    # 'PETS-S2l1',

    'SDD-coupa',
    'SDD-bookstore',
    'SDD-deathCircle',
    # 'SDD-gates',
    # 'SDD-hyang',
    # 'SDD-little',
    # 'SDD-nexus',
    # 'SDD-quad',

    # 'GC',

    'InD-1',  # location_id = 1
    'InD-2',  # location_id = 2
    # 'InD-3',  # location_id = 3
    # 'InD-4',  # location_id = 4

    'KITTI',
    'LCas-Minerva',
    'WildTrack',

    'Edinburgh',
    # 'Edinburgh-01Jul',
    # 'Edinburgh-01Aug',
    # 'Edinburgh-01Sep',

    # Bottleneck (Hermes)
    'BN-1d-w180',
    'BN-2d-w160',


    # 'TownCenter',
]


def get_trajlets(opentraj_root, dataset_names, to_numpy=True):
    trajlets = {}

    # Make a temp dir to store and load trajlets (no splitting anymore)
    trajlet_dir = os.path.join(opentraj_root, 'trajlets__temp_')
    if not os.path.exists(trajlet_dir): os.makedirs(trajlet_dir)
    for dataset_name in dataset_names:
        trajlet_npy_file = os.path.join(trajlet_dir, dataset_name + '-trl.npy')
        if to_numpy and os.path.exists(trajlet_npy_file):
            trajlets[dataset_name] = np.load(trajlet_npy_file)
            print("loading trajlets from: ", trajlet_npy_file)
        else:
            ds = get_datasets(opentraj_root, [dataset_name])[dataset_name]
            trajs = ds.get_trajectories(label="pedestrian")
            trajlets[dataset_name] = split_trajectories(trajs, to_numpy=to_numpy)
            if to_numpy:
                np.save(trajlet_npy_file, trajlets[dataset_name])
            print("writing trajlets ndarray into: ", trajlet_npy_file)

    return trajlets


# ================================================================
# ===================== Load the datasets ========================
# ================================================================
def get_datasets(opentraj_root, dataset_names, use_kalman='default'):
    datasets = {}

    # Make a temp dir to store and load trajdatasets (no postprocess anymore)
    trajdataset_dir = os.path.join(opentraj_root, 'trajdatasets__temp')
    if not os.path.exists(trajdataset_dir): os.makedirs(trajdataset_dir)

    for dataset_name in dataset_names:
        dataset_h5_file = os.path.join(trajdataset_dir, dataset_name + '.h5')
        if os.path.exists(dataset_h5_file):
            datasets[dataset_name] = TrajDataset()
            datasets[dataset_name].data = pd.read_pickle(dataset_h5_file)
            datasets[dataset_name].title = dataset_name
            print("loading dataset from pre-processed file: ", dataset_h5_file)
            continue

        print("Loading dataset:", dataset_name)

        # ========== ETH ==============
        if 'eth-univ' == dataset_name.lower():
            eth_univ_root = os.path.join(opentraj_root, 'datasets/ETH/seq_eth/obsmat.txt')
            datasets[dataset_name] = load_eth(eth_univ_root, title=dataset_name, scene_id='Univ',
                                              use_kalman=True)

        elif 'eth-hotel' == dataset_name.lower():
            eth_hotel_root = os.path.join(opentraj_root, 'datasets/ETH/seq_hotel/obsmat.txt')
            datasets[dataset_name] = load_eth(eth_hotel_root, title=dataset_name, scene_id='Hotel')
        # ******************************

        # ========== UCY ==============
        elif 'ucy-zara' == dataset_name.lower():  # all 3 zara sequences
            zara01_dir = os.path.join(opentraj_root, 'datasets/UCY/zara01')
            zara02_dir = os.path.join(opentraj_root, 'datasets/UCY/zara02')
            zara03_dir = os.path.join(opentraj_root, 'datasets/UCY/zara03')
            zara_01_ds = load_crowds(zara01_dir + '/annotation.vsp',
                                     homog_file=zara01_dir + '/H.txt',
                                     scene_id='1', use_kalman=True)
            zara_02_ds = load_crowds(zara02_dir + '/annotation.vsp',
                                     homog_file=zara02_dir + '/H.txt',
                                     scene_id='2', use_kalman=True)
            zara_03_ds = load_crowds(zara03_dir + '/annotation.vsp',
                                     homog_file=zara03_dir + '/H.txt',
                                     scene_id='3', use_kalman=True)
            datasets[dataset_name] = merge_datasets([zara_01_ds, zara_02_ds, zara_03_ds], dataset_name)

        elif 'ucy-univ' == dataset_name.lower():  # all 3 sequences
            st001_dir = os.path.join(opentraj_root, 'datasets/UCY/students01')
            st003_dir = os.path.join(opentraj_root, 'datasets/UCY/students03')
            uni_ex_dir = os.path.join(opentraj_root, 'datasets/UCY/uni_examples')
            #st001_ds = load_Crowds(st001_dir + '/students001.txt',homog_file=st001_dir + '/H.txt',scene_id='1',use_kalman=True)

            st001_ds = load_crowds(st001_dir + '/annotation.vsp',
                                   homog_file=st003_dir + '/H.txt',
                                   scene_id='st001', use_kalman=True)

            st003_ds = load_crowds(st003_dir + '/annotation.vsp',
                                   homog_file=st003_dir + '/H.txt',
                                   scene_id='st003', use_kalman=True)
            uni_ex_ds = load_crowds(uni_ex_dir + '/annotation.vsp',
                                    homog_file=st003_dir + '/H.txt',
                                    scene_id='uni-ex', use_kalman=True)
            datasets[dataset_name] = merge_datasets([st001_ds, st003_ds, uni_ex_ds], dataset_name)

        elif 'ucy-zara1' == dataset_name.lower():
            zara01_root = os.path.join(opentraj_root, 'datasets/UCY/zara01/obsmat.txt')
            datasets[dataset_name] = load_eth(zara01_root, title=dataset_name)

        elif 'ucy-zara2' == dataset_name.lower():
            zara02_root = os.path.join(opentraj_root, 'datasets/UCY/zara02/obsmat.txt')
            datasets[dataset_name] = load_eth(zara02_root, title=dataset_name)

        elif 'ucy-univ3' == dataset_name.lower():
            students03_root = os.path.join(opentraj_root, 'datasets/UCY/students03/obsmat.txt')
            datasets[dataset_name] = load_eth(students03_root, title=dataset_name)
        # ******************************

        # ========== HERMES ==============
        elif 'bn' in dataset_name.lower().split('-'):
            [_, exp_flow, cor_size] = dataset_name.split('-')
            if exp_flow == '1d' and cor_size == 'w180':   # 'Bottleneck-udf-180'
                bottleneck_path = os.path.join(opentraj_root, 'datasets/HERMES/Corridor-1D/uo-180-180-120.txt')
            elif exp_flow == '2d' and cor_size == 'w160':  # 'Bottleneck-bdf-160'
                bottleneck_path = os.path.join(opentraj_root, "datasets/HERMES/Corridor-2D/bo-360-160-160.txt")
            else:
                "Unknown Bottleneck dataset!"
                continue
            datasets[dataset_name] = load_bottleneck(bottleneck_path, sampling_rate=6,
                                                     use_kalman=True,
                                                     title=dataset_name)
        # ******************************

        # ========== PETS ==============
        elif 'pets-s2l1' == dataset_name.lower():
            pets_root = os.path.join(opentraj_root, 'datasets/PETS-2009/data')
            datasets[dataset_name] = load_pets(os.path.join(pets_root, 'annotations/PETS2009-S2L1.xml'),  #Pat:was PETS2009-S2L2
                                               calib_path=os.path.join(pets_root, 'calibration/View_001.xml'),
                                               sampling_rate=2,
                                               title=dataset_name)
        # ******************************

        # ========== GC ==============
        elif 'gc' == dataset_name.lower():
            gc_root = os.path.join(opentraj_root, 'datasets/GC/Annotation')
            datasets[dataset_name] = load_gcs(gc_root, world_coord=True, title=dataset_name,
                                              use_kalman=True
                                              )
        # ******************************

        # ========== InD ==============
        elif 'ind-1' == dataset_name.lower():
            ind_root = os.path.join(opentraj_root, 'datasets/InD/inD-dataset-v1.0/data')
            file_ids = range(7, 17 + 1)  # location_id = 1
            ind_1_datasets = []
            for id in file_ids:
                dataset_i = load_ind(os.path.join(ind_root, '%02d_tracks.csv' % id),
                                     scene_id='1-%02d' %id,
                                     sampling_rate=10,
                                     use_kalman=True)
                ind_1_datasets.append(dataset_i)
            datasets[dataset_name] = merge_datasets(ind_1_datasets, new_title=dataset_name)

        elif 'ind-2' == dataset_name.lower():
            ind_root = os.path.join(opentraj_root, 'datasets/InD/inD-dataset-v1.0/data')
            file_ids = range(18, 29 + 1)  # location_id = 2
            ind_2_datasets = []
            for id in file_ids:
                dataset_i = load_ind(os.path.join(ind_root, '%02d_tracks.csv' % id),
                                     scene_id='1-%02d' % id,
                                     sampling_rate=10,
                                     use_kalman=True)
                ind_2_datasets.append(dataset_i)
            datasets[dataset_name] = merge_datasets(ind_2_datasets, new_title=dataset_name)

        elif 'ind-3' == dataset_name.lower():
            ind_root = os.path.join(opentraj_root, 'datasets/InD/inD-dataset-v1.0/data')
            file_ids = range(30, 32 + 1)  # location_id = 3
            ind_3_datasets = []
            for id in file_ids:
                dataset_i = load_ind(os.path.join(ind_root, '%02d_tracks.csv' % id),
                                     scene_id='1-%02d' % id,
                                     sampling_rate=10,
                                     use_kalman=True)
                ind_3_datasets.append(dataset_i)
            datasets[dataset_name] = merge_datasets(ind_3_datasets, new_title=dataset_name)

        elif 'ind-4' == dataset_name.lower():
            ind_root = os.path.join(opentraj_root, 'datasets/InD/inD-dataset-v1.0/data')
            file_ids = range(0, 6 + 1)  # location_id = 4
            ind_4_datasets = []
            for id in file_ids:
                dataset_i = load_ind(os.path.join(ind_root, '%02d_tracks.csv' % id),
                                     scene_id='1-%02d' % id,
                                     sampling_rate=10,
                                     use_kalman=True)
                ind_4_datasets.append(dataset_i)
            datasets[dataset_name] = merge_datasets(ind_4_datasets, new_title=dataset_name)
        # ******************************

        # ========== KITTI ==============
        elif 'kitti' == dataset_name.lower():
            kitti_root = os.path.join(opentraj_root, 'datasets/KITTI/data')
            datasets[dataset_name] = load_kitti(kitti_root, title=dataset_name,
                                                use_kalman=True,
                                                sampling_rate=4)  # FixMe: apparently original_fps = 2.5
        # ******************************

        # ========== L-CAS ==============
        elif 'lcas-minerva' == dataset_name.lower():
            lcas_root = os.path.join(opentraj_root, 'datasets/L-CAS/data')
            datasets[dataset_name] = load_lcas(lcas_root, title=dataset_name,
                                               use_kalman=True,
                                               sampling_rate=1)  # FixMe: apparently original_fps = 2.5
        # ******************************

        # ========== Wild-Track ==============
        elif 'wildtrack' == dataset_name.lower():
            wildtrack_root = os.path.join(opentraj_root, 'datasets/Wild-Track/annotations_positions')
            datasets[dataset_name] = load_wildtrack(wildtrack_root, title=dataset_name,
                                                    use_kalman=True,
                                                    sampling_rate=1)  # original_annot_framerate=2
        # ******************************

        # ========== Edinburgh ==============
        elif 'edinburgh' in dataset_name.lower():
            edinburgh_dir = os.path.join(opentraj_root, 'datasets/Edinburgh/annotations')
            if 'edinburgh' == dataset_name.lower():   # all files
                # edinburgh_path = edinburgh_dir
                # select 1-10 Sep
                Ed_selected_days = ['01Sep', '02Sep', '04Sep', '05Sep', '06Sep', '10Sep']
                partial_ds = []
                for selected_day in Ed_selected_days:
                    edinburgh_path = os.path.join(edinburgh_dir, 'tracks.%s.txt' % selected_day)
                    partial_ds.append(load_edinburgh(edinburgh_path, title=dataset_name,
                                                     use_kalman=True, scene_id=selected_day,
                                                     sampling_rate=4)  # original_framerate=9
                                      )
                merge_datasets(partial_ds)

            else:
                seq_date = dataset_name.split('-')[1]
                edinburgh_path = os.path.join(edinburgh_dir, 'tracks.%s.txt' %seq_date)
            datasets[dataset_name] = load_edinburgh(edinburgh_path, title=dataset_name,
                                                    use_kalman=True,
                                                    sampling_rate=4)  # original_framerate=9
        # ******************************

        # ========== Town-Center ==============
        elif 'towncenter' == dataset_name.lower():
            towncenter_root = os.path.join(opentraj_root, 'datasets/Town-Center')
            # FixMe: might need Kalman Smoother
            datasets[dataset_name] = load_town_center(towncenter_root + '/TownCentre-groundtruth-top.txt',
                                                      calib_path=towncenter_root + '/TownCentre-calibration-ci.txt',
                                                      title=dataset_name,
                                                      use_kalman=True,
                                                      sampling_rate=10)  # original_framerate=25
            # ******************************

        # ========== SDD ==============
        elif 'sdd-' in dataset_name.lower():
            scene_name = dataset_name.split('-')[1]
            sdd_root = os.path.join(opentraj_root, 'datasets', 'SDD')
            annot_files_sdd = sorted(glob.glob(sdd_root + '/' + scene_name + "/**/annotations.txt", recursive=True))

            sdd_scales_yaml_file = os.path.join(sdd_root, 'estimated_scales.yaml')
            with open(sdd_scales_yaml_file, 'r') as f:
                scales_yaml_content = yaml.load(f, Loader=yaml.FullLoader)

            scene_datasets = []
            for file_name in annot_files_sdd:
                filename_parts = file_name.split('/')
                scene_name = filename_parts[-3]
                scene_video_id = filename_parts[-2]
                scale = scales_yaml_content[scene_name][scene_video_id]['scale']
                sdd_dataset_i = load_sdd(file_name, scale=scale,
                                         scene_id=scene_name + scene_video_id.replace('video', ''),
                                         drop_lost_frames=False,
                                         use_kalman=True,
                                         sampling_rate=12)  # original_framerate=30
                scene_datasets.append(sdd_dataset_i)
            scene_dataset = merge_datasets(scene_datasets, dataset_name)
            datasets[dataset_name] = scene_dataset
        # ******************************

        else:
            print("Error! invalid dataset name:", dataset_name)

        # save to h5 file
        datasets[dataset_name].data.to_pickle(dataset_h5_file)
        print("saving dataset into pre-processed file: ", dataset_h5_file)

    return datasets
