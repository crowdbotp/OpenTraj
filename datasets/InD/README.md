# The Intersection Drone Dataset 
Naturalistic Trajectories of Vehicles and Vulnerable Road Users Recorded at German Intersections

<p align='center'>
  <img src='reference.png' width='640px'\>  
</p>

## About the Dataset

The inD dataset is a new dataset of naturalistic vehicle trajectories recorded at German intersections. Using a drone, typical limitations of established traffic data collection methods like occlusions are overcome. Traffic was recorded at four different locations. The trajectory for each road user and its type is extracted. Using state-of-the-art computer vision algorithms, the positional error is typically less than 10 centimetres. The dataset is applicable on many tasks such as road user prediction, driver modeling, scenario-based safety validation of automated driving systems or data-driven development of HAD system components.

* All Types of Road Users: The dataset includes:
  - Vehicles
  - Pedestrians
  - Bicyclists

* High Quality and Variety: The dataset features:
  - Four different recording locations
  - Different intersection types
  - Typical positioning error <10 cm

* Easy Start: Provided scripts for Python: https://github.com/ika-rwth-aachen/drone-dataset-tools
  - Parising of provided files
  - Visualization of recorded trajectories

## Load Dataset with Toolkit
In order to the load the datasets, we provided the [`loader_ind.py`](../../toolkit/loaders/loader_ind.py)

```python
import os
from toolkit.loaders.loader_ind import load_ind
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
ind_root = os.path.join(OPENTRAJ_ROOT, 'datasets/InD/inD-dataset-v1.0/data')
file_id = 0 # range(0, 33)
traj_dataset = load_ind(os.path.join(ind_root, '%02d_tracks.csv' % file_id),
                        scene_id='1-%02d' %file_id, sampling_rate=1, use_kalman=True)
```
* Note: there are 4 different locations with totally 33 annotation_files (xx_tracks.csv):
  - location_id(1) => xx : 07 to 17 
  - location_id(2) => xx : 18 to 29 
  - location_id(3) => xx : 30 to 32 
  - location_id(4) => xx : 00 to 06 

## License
This dataset is free for non-commercial use only. However we (OpenTraj) do not have the permission to share it with you. You can directly contact the authors to get a access to the dataset: https://www.ind-dataset.com/

## Citation
```
@inproceedings{inDdataset,
               title={The inD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories at German Intersections},
               author={Bock, Julian and Krajewski, Robert and Moers, Tobias and Vater, Lennart and Runde, Steffen and Eckstein, Lutz},
               journal={arXiv preprint arXiv:1911.07602},
               year={2019}}
```
