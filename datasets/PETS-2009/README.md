# PETS-2009 Dataset
This [dataset](http://www.cvg.reading.ac.uk/PETS2009/a.html) is a set of multisensor sequences containing different crowd activities.

<p align='center'>
  <img src='reference.jpg' width=360 \>
</p>

The bounding-box annotations are provided by [milanton.de](http://www.milanton.de/data/)

The following sequences are included from the mian dataset:

| Scenario | Level  | Activity  | Crowd Density | Annotation Files |
| ----     | :---:  | :---:     | :---: | :---: |
| S1       | L1     | walking   | medium density crowd | `PETS2009-S1L1-1.xml` `PETS2009-S1L1-2.xml` |
| S1       | L2     | walking   | high density crowd | `PETS2009-S1L2-1.xml` `PETS2009-S1L2-2.xml` |
| S2       | L1     | walking   | sparse crowd | `PETS2009-S2L1.xml` |
| S2       | L1     | walking   | medium density crowd | `PETS2009-S2L2.xml` |
| S2       | L3     | walking   | dense crowd | `PETS2009-S2L3.xml` |
| S3       | Mult-flow | running   | dense crowd | `PETS2009-S3MF1.xml` |


## Annotations
All annotations were done by manually placing bounding boxes around pedestrians and interpolating their trajectories between key frames. The trajectories were then smoothed to avoid discontinuous or jittery movements. All targets were annotated in all sequences, even in case of total occlusion. Each person entering the field of view acquires a unique ID, i.e. if a person leaves the screen and later reappears again, a new ID is assigned. 
* **Note**: bounding boxes are not always perfectly aligned due to articulation, interpolation and mistakes made by the annotator.
The annotations are saved in xml files using the CVML Specification. The hierarchy is like:

```xml
<dataset name=FILENAME>
  <frame number=FRAME_NUMBER>
    <objectlist>
      <object id=OBJECTNUMBER>        
        <box h=HEIGHT w=WIDTH xc=XCENTER yc=YCENTER/>  
      </object>
      .........        Other objects            
    </objectlist>
  </frame>
  .........        Other frames      
</dataset>
```

* **Note**: For each dataset there are two ground truth files. One (complete) contains annotations for all visible targets, the other one (cropped) only contains targets within the predefined tracking area used in our experiments.

## Calibration
The annotations are done in View001 which is stored in [`calibration/View_001.xml`](./data/calibration/View_001.xml) and include the extrinsic and intrinsic parameters of the camera.

We provided [`camera_calibration_tsai.py`](../../toolkit/loaders/utils/camera_calibration_tsai.py) for transforming the trajectories from image space to world coordinate space. This code works based on calibration parameters defined in [Tsai camera calibration](https://www.dca.ufrn.br/~lmarcos/courses/visao/artigos/CameraCalibrationTsai.pdf).

## Load dataset with Toolkit
In order to the load the datasets, we provided the [`loader_pets.py`](../../toolkit/loaders/loader_pets.py)

```python
import os
from toolkit.loaders.loader_pets import load_pets
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
pets_root = os.path.join(OPENTRAJ_ROOT, 'datasets/PETS-2009/data')
datasets = load_pets(os.path.join(pets_root, 'annotations/PETS2009-S2L1.xml'),  
                     calib_path=os.path.join(pets_root, 'calibration/View_001.xml'),
                     sampling_rate=2, title='PETS-2009')
```


## License
Legal note by authors:
> The video sequences are copyright University of Reading and permission is hereby granted for free download for the purposes of the PETS 2009 workshop and academic and industrial research. Where the data is disseminated (e.g. publications, presentations) the source should be acknowledged.

## Citation
```
@inproceedings{ferryman2009pets2009,
  title={Pets2009: Dataset and challenge},
  author={Ferryman, James and Shahrokni, Ali},
  booktitle={2009 Twelfth IEEE international workshop on performance evaluation of tracking and surveillance},
  pages={1--6},
  year={2009},
  organization={IEEE}
}
```
