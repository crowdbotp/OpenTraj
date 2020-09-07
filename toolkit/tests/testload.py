from crowdscan.loader.loader_eth import loadETH
from crowdscan.loader.loader_gc import loadGC
from crowdscan.loader.loader_sdd import loadSDD_single, loadSDD_all
from crowdscan.loader.loader_pets import loadPETS
from crowdscan.loader.loader_ind import load_ind
from crowdscan.loader.loader_wildtrack import loadWildTrack
from crowdscan.loader import loader_ind
from crowdscan.loader.loader_town import loadTownCenter
import sys
import os


def run(path, args):

    print("\n-----------------------------\nRunning test load\n-----------------------------")
    if 'eth/' in path.lower():
        print("[Javad]: Directly reading ETH Dataset (seq_eth):")
        traj_dataset = loadETH(path)
        all_trajs = traj_dataset.get_trajectories()
        all_frames = traj_dataset.get_frames()

    if '/sdd' in path.lower():
        if os.path.isdir(path):
            traj_dataset = loadSDD_all(path)
        else:
            traj_dataset = loadSDD_single(path)
        trajs = traj_dataset.get_trajectories()
        print("total number of trajectories = ", len(trajs))

    if 'gc/' in path.lower():
        kwargs = {}
        for arg in args:
            if 'homog_file=' in arg:
                kwargs['homog_file'] = arg.replace("homog_file=", "")
        gc_dataset = loadGC(path, **kwargs)
        trajs = gc_dataset.get_trajectories()
        print("GC: number of trajs = ", len(trajs))

    if 'pets-2009/' in path.lower():
        kwargs = {}
        for arg in args:
            if 'calib_path=' in arg:
                kwargs['calib_path'] = arg.replace("calib_path=", "")
        loadPETS(path, **kwargs)

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
        traj_dataset = loadWildTrack(path)

    if 'town' in path.lower():
        # Construct arguments dictionary
        kwargs = {}
        for arg in args:
            if 'calib_path=' in arg:
                kwargs['calib_path'] = arg.replace("calib_path=", "")

        # Test the Town Center Dataset
        traj_dataset = loadTownCenter(path, **kwargs)
        all_trajs = traj_dataset.get_trajectories()
        print('------------------------')
        print('First trajectory (Town Center)')
        print('------------------------')
        print(all_trajs[0])
        all_frames = traj_dataset.get_frames()

    if 'chaos' in path.lower():
        print("\n")
        print("ChAOS Style :")
        print(loader.loadChAOS(path, args.separator))

        print("\n\n-----------------------------\nTest load done\n-----------------------------")


if __name__ == '__main__':
    path = sys.argv[1]
    kwargs = []
    if len(sys.argv) > 2:
        kwargs = sys.argv[2:]
    run(path, kwargs)
