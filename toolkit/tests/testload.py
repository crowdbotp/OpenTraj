from toolkit.loaders.loader_eth import load_eth
from toolkit.loaders.loader_gcs import load_gcs
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir
from toolkit.loaders.loader_pets import load_pets
from toolkit.loaders.loader_ind import load_ind
from toolkit.loaders.loader_wildtrack import load_wildtrack
from toolkit.loaders.loader_town import load_town_center
import sys
import os


def run(path, args):

    print("\n-----------------------------\nRunning test load\n-----------------------------")
    if 'eth/' in path.lower():
        print("[Javad]: Directly reading ETH Dataset (seq_eth):")
        traj_dataset = load_eth(path)
        all_trajs = traj_dataset.get_trajectories()
        all_frames = traj_dataset.get_frames()

    if '/sdd' in path.lower():
        if os.path.isdir(path):
            traj_dataset = load_sdd_dir(path)
        else:
            traj_dataset = load_sdd(path)
        trajs = traj_dataset.get_trajectories()
        print("total number of trajectories = ", len(trajs))

    if 'gc/' in path.lower():
        kwargs = {}
        for arg in args:
            if 'homog_file=' in arg:
                kwargs['homog_file'] = arg.replace("homog_file=", "")
        gc_dataset = load_gcs(path, **kwargs)
        trajs = gc_dataset.get_trajectories()
        print("GC: number of trajs = ", len(trajs))

    if 'pets-2009/' in path.lower():
        kwargs = {}
        for arg in args:
            if 'calib_path=' in arg:
                kwargs['calib_path'] = arg.replace("calib_path=", "")
        load_pets(path, **kwargs)

    if 'ind/' in path.lower():
        # Test the InD Dataset
        traj_dataset = load_ind(path)
        all_trajs = traj_dataset.get_trajectories()
        print('------------------------')
        print('First trajectory (InD)')
        print('------------------------')
        print(all_trajs[0])
        all_frames = traj_dataset.get_frames()

    if 'wild-track/' in path.lower():
        traj_dataset = load_wildtrack(path)

    if 'town' in path.lower():
        # Construct arguments dictionary
        kwargs = {}
        for arg in args:
            if 'calib_path=' in arg:
                kwargs['calib_path'] = arg.replace("calib_path=", "")

        # Test the Town Center Dataset
        traj_dataset = load_town_center(path, **kwargs)
        all_trajs = traj_dataset.get_trajectories()
        print('------------------------')
        print('First trajectory (Town Center)')
        print('------------------------')
        print(all_trajs[0])
        all_frames = traj_dataset.get_frames()

    if 'chaos' in path.lower():
        print("\n")
        print("ChAOS Style :")
        print(loaders.loadChAOS(path, args.separator))

        print("\n\n-----------------------------\nTest load done\n-----------------------------")


if __name__ == '__main__':
    path = sys.argv[1]
    kwargs = []
    if len(sys.argv) > 2:
        kwargs = sys.argv[2:]
    run(path, kwargs)
