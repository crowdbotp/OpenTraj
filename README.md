# OpenTraj
Trajectory Prediction Benchmark and State-of-the-art

## Public Available Trajectory Datasets

<!--begin(table_main)-->
| Sample | Name | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Description&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Ref | 
|----|----|----|----|
| ![](ETH/seq_eth/reference.png) | [ETH](https://github.com/amiryanj/OpenTraj/blob/master/ETH) | 2 top view scenes containing walking pedestrians <code>#Traj:[Peds=750]</code> <code>Coord=world-2D</code> <code>FPS=2.5</code> | [website](http://www.vision.ee.ethz.ch/en/datasets/) [paper](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/pellegrini09iccv.pdf) | 
| ![](UCY/zara01/reference.png) | [UCY](https://github.com/amiryanj/OpenTraj/blob/master/UCY) | 3 scenes (Zara/Arxiepiskopi/University). Zara and University close to top view. Arxiepiskopi more inclined. <code>#Traj:[Peds=786]</code> <code>Coord=world-2D</code> <code>FPS=2.5</code> | [website](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) [paper](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2007.01089.x) | 
| ![](PETS-2009/reference.png) | [PETS 2009](https://github.com/amiryanj/OpenTraj/blob/master/PETS-2009) | different crowd activities <code>#Traj:[?]</code> <code>Coord=image-2D</code> <code>FPS=7</code> | [website](http://www.cvg.reading.ac.uk/PETS2009/data.html) [paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/AVSS-2010/data/4264a143.pdf) | 
| ![](SDD/coupa/video3/reference.jpg) | [SDD](https://github.com/amiryanj/OpenTraj/blob/master/SDD) | 8 top view scenes recorded by drone contains various types of agents <code>#Traj:[Bikes=4210 Peds=5232 Skates=292 Carts=174 Cars=316 Buss=76 Total=10,300]</code> <code>Coord=image-2D</code> <code>FPS=30</code> | [website](http://cvgl.stanford.edu/projects/uav_data) [paper](http://svl.stanford.edu/assets/papers/ECCV16social.pdf) | 
| ![](GC/reference.jpg) | [GC](https://github.com/amiryanj/OpenTraj/blob/master/GC) | Grand Central Train Station Dataset: 1 scene of 33:20 minutes of crowd trajectories <code>#Traj:[Peds=12,684]</code> <code>Coord=image-2D</code> <code>FPS=25</code> | [dropbox](https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar) [paper](http://openaccess.thecvf.com/content_cvpr_2015/html/Yi_Understanding_Pedestrian_Behaviors_2015_CVPR_paper.html) | 
| ![](HERMES/reference.png) | [HERMES](https://github.com/amiryanj/OpenTraj/blob/master/HERMES) | Controlled Experiments of Pedestrian Dynamics (Unidirectional and bidirectional flows) <code>#Traj:[?]</code> <code>Coord=world-2D</code> <code>FPS=16</code> | [website](https://www.fz-juelich.de/ias/ias-7/EN/AboutUs/Projects/Hermes/_node.html) [data](https://www.fz-juelich.de/ias/ias-7/EN/Research/Pedestrian_Dynamics-Empiricism/_node.html) | 
| ![](Waymo/reference.jpg) | [Waymo](https://github.com/amiryanj/OpenTraj/blob/master/Waymo) | High-resolution sensor data collected by Waymo self-driving cars <code>#Traj:[?]</code> <code>Coord=2D and 3D</code> <code>FPS=?</code> | [website](https://waymo.com/open/) [github](https://github.com/waymo-research/waymo-open-dataset) | 
| ![](KITTI/reference.jpg) | [KITTI](https://github.com/amiryanj/OpenTraj/blob/master/KITTI) | 6 hours of traffic scenarios. various sensors <code>#Traj:[?]</code> <code>Coord=image-3D + Calib</code> <code>FPS=10</code> |  [website](http://www.cvlibs.net/datasets/kitti/) | 
| ![](InD/reference.png) | [inD](https://github.com/amiryanj/OpenTraj/blob/master/InD) | Naturalistic Trajectories of Vehicles and Vulnerable Road Users Recorded at German Intersections <code>#Traj:[Total=11,500]</code> <code>Coord=world-2D</code> <code>FPS=25</code> | [website](https://www.ind-dataset.com/) [paper](https://arxiv.org/pdf/1911.07602.pdf) | 
| ![](L-CAS/reference.png) | [L-CAS](https://github.com/amiryanj/OpenTraj/blob/master/L-CAS) | Multisensor People Dataset Collected by a Pioneer 3-AT robot <code>#Traj:[?]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](https://lcas.lincoln.ac.uk/wp/research/data-sets-software/l-cas-multisensor-people-dataset/) | 
| ![](VIRAT/reference.png) | [VIRAT](https://github.com/amiryanj/OpenTraj/blob/master/VIRAT) | Natural scenes showing people performing normal actions <code>#Traj:[?]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](http://viratdata.org/) | 
| ![](VRU/reference.png) | [VRU](https://github.com/amiryanj/OpenTraj/blob/master/VRU) | consists of pedestrian and cyclist trajectories, recorded at an urban intersection using cameras and LiDARs <code>#Traj:[peds=1068 Bikes=464]</code> <code>Coord=World (Meter)</code> <code>FPS=25</code> | [website](https://www.th-ab.de/ueber-uns/organisation/labor/kooperative-automatisierte-verkehrssysteme/trajectory-dataset) | 
| ![](Edinburgh/reference.jpg) | [Edinburgh](https://github.com/amiryanj/OpenTraj/blob/master/Edinburgh) | People walking through the Informatics Forum (University of Edinburgh) <code>#Traj:[ped=+92,000]</code> <code>FPS=0</code> | [website](http://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/) | 
| ![](Town-Center/reference.jpg) | [Town Center](https://github.com/amiryanj/OpenTraj/blob/master/Town-Center) | CCTV video of pedestrians in a busy downtown area in Oxford <code>#Traj:[peds=2,200]</code> <code>Coord=0</code> <code>FPS=0</code> | [website](https://megapixels.cc/datasets/oxford_town_centre/) | 
| ![](ATC/reference.png) | [ATC](https://github.com/amiryanj/OpenTraj/blob/master/ATC) | 92 days of pedestrian trajectories in a shopping center in Osaka, Japan <code>#Traj:[?]</code> <code>Coord=world-2D + Range data</code> | [website](https://irc.atr.jp/crest2010_HRI/ATC_dataset) | 
| ![](City-Scapes/reference.png) | [City Scapes](https://github.com/amiryanj/OpenTraj/blob/master/City-Scapes) | 25,000 annotated images (Semantic/ Instance-wise/ Dense pixel annotations) <code>#Traj:[?]</code> | [website](https://www.cityscapes-dataset.com/dataset-overview/) | 
| ![](Forking-Paths-Garden/reference.png) | [Forking Paths Garden](https://github.com/amiryanj/OpenTraj/blob/master/Forking-Paths-Garden) | **Multi-modal** _Synthetic_ dataset, created in [CARLA](https://carla.org) (3D simulator) based on real world trajectory data, extrapolated by human annotators <code>#Traj:[?]</code> | [website](https://next.cs.cmu.edu/multiverse/index.html) [github](https://github.com/JunweiLiang/Multiverse) [paper](https://arxiv.org/abs/1912.06445) | 
| ![](NuScenes/reference.png) | [nuScenes](https://github.com/amiryanj/OpenTraj/blob/master/NuScenes) | Large-scale Autonomous Driving dataset <code>#Traj:[peds=222,164 vehicles=662,856]</code> <code>Coord=World + 3D Range Data</code> <code>FPS=2</code> | [website](www.nuscences.org) | 
| ![](Argoverse/reference.jpg) | [Argoverse](https://github.com/amiryanj/OpenTraj/blob/master/Argoverse) | 320 hours of Self-driving dataset <code>#Traj:[objects=11,052]</code> <code>Coord=3D</code> <code>FPS=10</code> | [website](https://www.argoverse.org) | 
| ![](Wild-Track/reference.jpg) | [Wild Track](https://github.com/amiryanj/OpenTraj/blob/master/Wild-Track) | surveillance video dataset of students recorded outside the ETH university main building in Zurich. <code>#Traj:[peds=1,200]</code> | [website](https://megapixels.cc/wildtrack/) | 
| ![] | DUT | Natural Vehicle-Crowd Interactions in crowded university campus <code>#Traj:[Peds=1,739 vehicles=123 Total=1,862]</code> <code>Coord=world-2D</code> <code>FPS=23.98</code> | [github](https://github.com/dongfang-steven-yang/vci-dataset-citr) [paper](https://arxiv.org/pdf/1902.00487.pdf) | 
| ![] | CITR | Fundamental Vehicle-Crowd Interaction scenarios in controlled experiments <code>#Traj:[Peds=340]</code> <code>Coord=world-2D</code> <code>FPS=29.97</code> | [github](https://github.com/dongfang-steven-yang/vci-dataset-dut) [paper](https://arxiv.org/pdf/1902.00487.pdf) | 
| ![] | Ko-PER | Trajectories of People and vehicles at Urban Intersections (Laserscanner + Video) <code>#Traj:[peds=350]</code> <code>Coord=world-2D</code> | [paper](https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.110/Bilder/Forschung/Datensaetze/20141010_DatasetDocumentation.pdf) | 
| ![](TRAF/reference.png) | [TRAF](https://github.com/amiryanj/OpenTraj/blob/master/TRAF) | small dataset of dense and heterogeneous traffic videos in India (22 footages) <code>#Traj:[Cars=33 Bikes=20 Peds=11]</code> <code>Coord=image-2D</code> <code>FPS=10</code> | [website](https://gamma.umd.edu/researchdirections/autonomousdriving/trafdataset/) [gDrive](https://drive.google.com/drive/folders/1zKaeboslkqoLdTJbRMyQ0Y9JL3007LRr) [paper](https://arxiv.org/pdf/1812.04767.pdf) | 
|   | ETH-Person |  | [website](https://data.vision.ee.ethz.ch/cvl/aess/) | 

<!--end(table_main)-->


#### Other Trajectory Datasets
- [NGSim](https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories)
- [Daimler](http://www.gavrila.net/Datasets/Daimler_Pedestrian_Benchmark_D/daimler_pedestrian_benchmark_d.html)
- [Cyclist](No Link)
- [highD](No Link)


#### Human Trajectory Prediction Benchmarks
- [Trajnet](http://trajnet.stanford.edu/): Trajectory Forecasting Challenge
- [MOT-Challenge](https://motchallenge.net): Multiple Object Tracking Benchmark

## Tools
OpenTraj provids a set of tools to load, visualize and analyze the trajectory datasets. (So far few datasets are supported).
#### 1. Parser
Using python files in [parser](toolkit/parser) dir, you can load a dataset into a dataset object. This object then can be used to retrieve the trajectories, with different queries (by id, timestamp, ...).
#### 2. play.py
Using [play.py](toolkit/play.py) script you can visualize a specific dataset, in a basic graphical interface.

<p align='center'>
  <img src='doc/figs/fig-opentraj-ui.gif' width='400px'\>
</p>

## Metrics
**1. ADE** (T<sub>obs</sub>, T<sub>pred</sub>):
Average Displacement Error (ADE), also called Mean Euclidean Distance (MED), measures the averages Euclidean distances between points of the predicted trajectory and the ground truth that have the same temporal distance from their respective start points. The function arguemnts are:
- T<sub>obs</sub> : observation period
- T<sub>pred</sub> : prediction period

**2. FDE** (T<sub>obs</sub>, T<sub>pred</sub>):
Final Displacement Error (FDE) measures the distance between final predicted position and the ground truth position at the corresponding time point. The function arguemnts are:
- T<sub>obs</sub> : observation period
- T<sub>pred</sub> : prediction period




## State-of-the-art Trajectory Prediction Algorithms
\* The numbers are derived from papers.
- [ ] setup benchmarking 
- [ ] update top 20 papers

#### 1. ETH Dataset

<!--begin(table_ETH)-->
| Method | Univ (ADE/FDE)* | Hotel (ADE/FDE)* | REF | 
|----|----|----|----|
| [Social-Force]() | 0.67 / 1.52 | 0.52 / 1.03 | 0 | 
| [Social-LSTM]() | 1.09 / 2.35 | 0.79 / 1.76 | 0 | 
| [Social-GAN](github.com/agrimgupta92/sgan) | 0.77 / 1.38 | 0.70 / 1.43 | 0 | 

<!--end(table_ETH)-->

`TBC`

<!-- 
| [Social-Attention]() <sup>[REF](#references)</sup>                                  | ?  | ?  |
| [SoPhie]() <sup>[REF]()</sup>                                            | ?  | ?  |
| [CIDNN](github.com/svip-lab/CIDNN) <sup>[REF]()</sup>            | ?  | ?  |
| [Social-Etiquette]() <sup>[REF]()</sup>            | ?  | ?  |
| [ConstVel]() <sup>[REF]()</sup>            | ?  | ?  |
| [Scene-LSTM]() <sup>[REF]()</sup>            | ?  | ?  |
| [Peeking Into the Future]() <sup>[REF]()</sup>            | ?  | ?  |
| [SS-LSTM]() <sup>[REF]()</sup>            | ?  | ?  |
| [MX-LSTM]() <sup>[REF]()</sup>            | ?  | ?  |
| [Social-BiGAT]() <sup>[REF]()</sup>            | ?  | ?  |
| [SR-LSTM]() <sup>[REF]()</sup>            | ?  | ?  |
-->

&ast; The values are in meter, calculated with ADE(T<sub>obs</sub>=3.2<sub>s</sub>, T<sub>pred</sub>=4.8<sub>s</sub>) and FDE(T<sub>obs</sub>=3.2<sub>s</sub>, T<sub>pred</sub>=4.8<sub>s</sub>).
<!--% Social Force => (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5995468) -->
<!--% Social Attention => (https://www.ri.cmu.edu/wp-content/uploads/2018/08/main.pdf) -->

<!--
- [Social-Etiquette](https://infoscience.epfl.ch/record/230262/files/ECCV16social.pdf)
- [ConstVel(The simpler, the better)](https://arxiv.org/pdf/1903.07933)
- [Scene-LSTM](https://arxiv.org/pdf/1808.04018)
- [Peeking Into the Future](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liang_Peeking_Into_the_Future_Predicting_Future_Person_Activities_and_Locations_CVPR_2019_paper.pdf)
- [SS-LSTM](https://ieeexplore.ieee.org/iel7/8345804/8354104/08354239.pdf)
- [MX-LSTM](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hasan_MX-LSTM_Mixing_Tracklets_CVPR_2018_paper.pdf)
- [Social-BiGAT](http://papers.nips.cc/paper/8308-social-bigat-multimodal-trajectory-forecasting-using-bicycle-gan-and-graph-attention-networks.pdf)
- [SR-LSTM](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_SR-LSTM_State_Refinement_for_LSTM_Towards_Pedestrian_Trajectory_Prediction_CVPR_2019_paper.pdf)
-->

#### 2. UCY Dataset
`TBC`
<!--begin(table-UCY)-->
<!-- 
| Method                                              | ZARA01 (ADE/FDE) | ZARA02 (ADE/FDE) | Students (ADE/FDE) |
| ------------------------------------------------------------------------------ | -- | -- | -- |
| [Social-Force]() <sup>[1](#references)</sup>                                   | ?  | ?  | ?  |
| [Social-Etiquette]() <sup>[REF]()</sup>                                        | ?  | ?  | ?  |
| [Social-LSTM]() <sup>[2](#references)</sup>                                    | ?  | ?  | ?  |
| [Social-GAN](github.com/agrimgupta92/sgan) <sup>[REF](#references)</sup>       | ?  | ?  | ?  |
| [CIDNN](github.com/svip-lab/CIDNN) <sup>[REF]()</sup>                          | ?  | ?  | ?  |
| [Social-Attention]() <sup>[REF](#references)</sup>                             | ?  | ?  | ?  |
| [Scene-LSTM]() <sup>[REF]()</sup>                                              | ?  | ?  | ?  |
| [ConstVel]() <sup>[REF]()</sup>                                                | ?  | ?  | ?  |
| [SoPhie]() <sup>[REF]()</sup>                                                  | ?  | ?  | ?  |
| [Social-Ways](github.com/amiryanj/socialways) <sup>[REF](#references)</sup>    | ?  | ?  | ?  |
| [Peeking Into the Future]() <sup>[REF]()</sup>                                 | ?  | ?  | ?  |
| [SS-LSTM]() <sup>[REF]()</sup>                                                 | ?  | ?  | ?  |
| [Social-BiGAT]() <sup>[REF]()</sup>                                            | ?  | ?  | ?  |
| [SR-LSTM]() <sup>[REF]()</sup>                                                 | ?  | ?  | ?  |
-->
<!--end(table-UCY)-->
#### 3. Other Datasets
- Stanford Drone Dataset (SDD)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [Social-Etiquette](https://infoscience.epfl.ch/record/230262/files/ECCV16social.pdf)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [DESIRE](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_DESIRE_Distant_Future_CVPR_2017_paper.pdf)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [SoPhie](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sadeghian_SoPhie_An_Attentive_GAN_for_Predicting_Paths_Compliant_to_Social_CVPR_2019_paper.pdf)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [MATF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Multi-Agent_Tensor_Fusion_for_Contextual_Trajectory_Prediction_CVPR_2019_paper.pdf)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [Best of Many](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Accurate_and_Diverse_CVPR_2018_paper.pdf)

- Grand Central Station (GC):

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [CIDNN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Encoding_Crowd_Interaction_CVPR_2018_paper.pdf)

- KITI

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; :small_blue_diamond: [R2P2](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nicholas_Rhinehart_R2P2_A_ReparameteRized_ECCV_2018_paper.pdf)

## Collaboration
Are you interested in collaboration on OpenTraj? Send an email to [me](mailto:amiryan.j@gmail.com?subject=OpenTraj) titled *OpenTraj*.

## References
#### (A) Main References:
- Who are you with and Where are you going? (Social Force), Yamaguchi et al. CVPR 2011. [paper]()
- Social LSTM: Human trajectory prediction in crowded spaces, Alahi et al. CVPR 2016. [paepr]()
- Learning social etiquette: Human trajectory understanding in crowded scenes, Robicquet et al. ECCV 2016. [paper](https://infoscience.epfl.ch/record/230262/files/ECCV16social.pdf) 
- Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks, Gupta et al. CVPR 2018. [paper]()
- Social Ways: Learning Multi-Modal Distributions of Pedestrian Trajectories with GANs, Amirian et al. CVPR 2019. [paper](), [code]()

** A more complete list of references can be found [here](https://github.com/jiachenli94/Awesome-Interaction-aware-Trajectory-Prediction)
<!--
- Desire: Distant future prediction in dynamic scenes with interacting agents, Lee et al. CVPR 2017. [paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lee_DESIRE_Distant_Future_CVPR_2017_paper.pdf)
- Sophie: An attentive gan for predicting paths compliant to social and physical constraints, Sadeghian et al. CVPR 2019. [paper](https://arxiv.org/pdf/1806.01482.pdf)
- [MATF (Multi-Agent Tensor Fusion)](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Multi-Agent_Tensor_Fusion_for_Contextual_Trajectory_Prediction_CVPR_2019_paper.pdf)
- [Best of Many](http://openaccess.thecvf.com/content_cvpr_2018/papers/Bhattacharyya_Accurate_and_Diverse_CVPR_2018_paper.pdf)
-->

#### (B) Surveys:
&ast; ordered by time
- A Survey on Path Prediction Techniques for Vulnerable Road Users: From Traditional to Deep-Learning Approaches, ITSC 2019. [paper](https://ieeexplore.ieee.org/abstract/document/8917053)
- Human Motion Trajectory Prediction: A Survey, IJRR 2019 [arxiv](https://arxiv.org/abs/1905.06113)
- Autonomous vehicles that interact with pedestrians: A survey of theory and practice, ITS 2019. [arxiv](https://arxiv.org/abs/1805.11773)
- A literature review on the prediction of pedestrian behavior in urban scenarios, ITSC 2018. [paper](https://ieeexplore.ieee.org/abstract/document/8569415)
- Survey on Vision-Based Path Prediction, DAPI 2018. [arxiv](https://arxiv.org/abs/1811.00233)
- Trajectory data mining: an overview, TIST 2015. [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2015/09/TrajectoryDataMining-tist-yuzheng.pdf)
- A survey on motion prediction and risk assessment for intelligent vehicles, ROBOMECH 2014. [paper](https://core.ac.uk/download/pdf/81530180.pdf)


