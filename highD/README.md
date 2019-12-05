# The Highway Drone Dataset
Naturalistic Trajectories of 110 500 Vehicles Recorded at German Highways

![](reference.png)

## About the Dataset

The highD dataset is a new dataset of naturalistic vehicle trajectories recorded on German highways. Using a drone, typical limitations of established traffic data collection methods such as occlusions are overcome by the aerial perspective. Traffic was recorded at six different locations and includes more than 110 500 vehicles. Each vehicle's trajectory, including vehicle type, size and manoeuvres, is automatically extracted. Using state-of-the-art computer vision algorithms, the positioning error is typically less than ten centimeters. Although the dataset was created for the safety validation of highly automated vehicles, it is also suitable for many other tasks such as the analysis of traffic patterns or the parameterization of driver models. Click [here](https://www.highd-dataset.com/details) for details.


## Large-scale Dataset

The dataset includes:
- 110,500 vehicles
- 44,500 driven kilometers
- 147 driven hours


## High Quality and Variety

The dataset features:
- Six different recording locations
- Different traffic states (e.g. traffic jams)
- Typical positioning error <10 cm


## Enriched Data

Pre-extracted information include:
- Surrounding vehicles
- Metrics like THW or TTC
- Driven maneuvers (e.g. lane changes)


## Easy Start

Provided scripts for Matlab and Python:
- Visualization of recorded trajectories
- Maneuver classification (soon)
- Maneuver statistics (soon)


## Citation
Our paper introducing the dataset and the used methods is published at the [IEEE ITSC 2018](https://www.ieee-itsc2018.org/). A preprint on arXiv.org is available [here](https://arxiv.org/abs/1810.05642). To reference the dataset, please cite this publication:

```
@inproceedings{highDdataset,
               title={The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems},
               author={Krajewski, Robert and Bock, Julian and Kloeker, Laurent and Eckstein, Lutz},
               booktitle={2018 21st International Conference on Intelligent Transportation Systems (ITSC)},
               pages={2118-2125},
               year={2018},
               doi={10.1109/ITSC.2018.8569552}}
```
