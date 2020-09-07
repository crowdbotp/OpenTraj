import sys
from toolkit.core.trajdataset import TrajDataset
from toolkit.core.trajlet import split_trajectories


# TODO:
def grouping(dataset: TrajDataset):
    trajs = dataset.get_trajectories("pedestrian")
    trajlets = split_trajectories(trajs)
    for trajlet in trajlets:
        pass
    return


if __name__ == "__main__":
    from toolkit.benchmarking.load_all_datasets import get_datasets
    opentraj_root = sys.argv[1]  # e.g. os.path.expanduser("~") + '/workspace2/OpenTraj'
    output_dir = sys.argv[2]  # e.g. os.path.expanduser("~") + '/Dropbox/OpenTraj-paper/exp/ver-0.2'
    datasets = get_datasets(opentraj_root)
    for ds in datasets:
        grouping(ds)
