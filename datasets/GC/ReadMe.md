# Grand Central Station Dataset
This dataset is collected from Grand Central Station in New York:
<p align='center'>
  <img src='reference.jpg' width='400px'\>  
</p>

* Link: https://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html
* Full dataset: https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar

| Details |  |
|-------------------------------------	| :---:	|
| Resolution (pixel)         	          | 1,920 × 1,080 	|
| Total frame number               	    | 100,000 	|
| Frame rate (fps)                      | 25  	|
| Annotated frame number             	  | 5,000  	|
| Annotated frame rate (fps) 	          | 1.25 (or 1.2) 	|
| Annotated pedestrian number       	  | 12,684 	|
| Average pedestrian number per frame  	| 123 	|
| Max pedestrian number per frame     	| 332  	|


## Homography
We manullay calculated a homography matrix: [`H.json`](./H.json)
based on the following information: **"The Main Concourse, is located on the upper platform level of Grand Central, in the geographical center of the station building. The cavernous concourse measures 275 ft (84 m) long by 120 ft (37 m) wide by 125 ft (38 m) high."** (The plan of the station is shown below):
* _Schlichting, Kurt C. (2001). Grand Central Terminal: Railroads, Architecture and Engineering in New York. Baltimore: Johns Hopkins University Press. ISBN 978-0-8018-6510-7_
<p align='center'>
  <img src='plan.png' width='400px'\>    
</p>
  
## Load Dataset with Toolkit

```python
import sys, os
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
path = os.path.join({OPENTRAJ_ROOT}, "datasets/GC/Annotation")
traj_dataset = load_gcs(path)
```

## License 
No license is issued with this dataset.

### Citation
The videos were originally collected by the authors of the following paper:
```
@inproceedings{zhou2011random,
  title={Random field topic model for semantic region analysis in crowded scenes from tracklets},
  author={Zhou, Bolei and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={CVPR 2011},
  pages={3441--3448},
  year={2011},
  organization={IEEE}
}
```
And later the trajectories dataset were created by the authors of the following work:
```
@inproceedings{yi2015understanding,
  title={Understanding pedestrian behaviors from stationary crowd groups},
  author={Yi, Shuai and Li, Hongsheng and Wang, Xiaogang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3488--3496},
  year={2015}
}
```
