# OpenTraj
Trajectory Prediction Benchmark and State-of-the-art


## Table of Public Available Trajectory Datasets

| Sample	                           | Name |	\#Trajs | Coord | FPS |	Density |	\*\*\*\*\*\*Description\*\*\*\*\*\* |	REF |
| ---------------------------------- | ---- | -------- | ----- | --- | -------- | ----- | ---- |
| ![](ETH/seq_eth/reference.png)     | ETH  | 750      | world | 2.5 | ?        |       | [website](http://www.vision.ee.ethz.ch/~stefpell/lta/index.html) [paper](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/pellegrini09iccv.pdf)| 
| ![](UCY/data_zara01/reference.png) | UCY  | 786      | world | 2.5 | ?        |       | [website](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) [paper](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2007.01089.x)| 
| ![](SDD/coupa/video3/reference.jpg)| SDD  | x Pedestrian x Bicyclist x Skateboarder	x Cart	x Car	x Bus Total = xxx  | image |     | ?        |       | [website](http://cvgl.stanford.edu/projects/uav_data) [paper](http://svl.stanford.edu/assets/papers/ECCV16social.pdf)|
| ![](GC/reference.png)              | GC   | 12,684   | image |     | ?        |       | [dropbox](https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar) [paper](http://openaccess.thecvf.com/content_cvpr_2015/html/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.html)|


## Metrics
**1. ADE** _[To, Tp]_

**2. FDE** _[To, Tp]_

## State-of-the-arts Trajectory Prediction Algorithms
#### A. ETH Dataset
| Method	         | ETH (ADE*/FDE*) |	Hotel (ADE/FDE) |
| ---------------- | --------------- | ---------------- |
| [Social-LSTM]()  | ? | ? |
| [Social-GAN]()   | ? | ? |
| [Social-Ways]()  | ? | ? |
| [SoPhie]()       | ? | ? |
| [CIDNN]()        | ? | ? |

#### B. UCY Dataset
| Method           | ZARA01 (ADE/FDE) | ZARA02 (ADE/FDE) | Students (ADE/FDE) |
| ---------------- | ---------------- | ---------------- | ------------------ |
| [Social-LSTM]()  | ? | ? | ? |
| [Social-GAN]()   | ? | ? | ? |
| [Social-Ways]()  | ? | ? | ? |
| [SoPhie]()       | ? | ? | ? |
| [CIDNN]()        | ? | ? | ? |

- ConstVel
- SS-LSTM

#### C. Stanford Drone Dataset (SDD)
1. DESIRE
2. SoPhie

#### D. Grand Central Station (GC) Dataset
1. CIDNN

#### REF
1. MOT Challenge
2. Trajnet


