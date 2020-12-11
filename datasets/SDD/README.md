# Stanford Drone Dataset
The Stanford Aerial Pedestrian Dataset consists of annotated videos of pedestrians, bikers, skateboarders, cars, buses, and golf carts navigating eight unique scenes on the Stanford University campus: https://cvgl.stanford.edu/projects/uav_data/

<p align='center'>
  <img src='./hyang/video6/reference.jpg' width='480px'\>
</p>

## Scenes
The eight scenes are:

- bookstore
- coupa
- deathCircle
- gates
- hyang
- little
- nexus
- quad

Each video for each scene in the videos directory has an associated annotation file (annotation.txt) and exemplary frame (reference.jpg) in the annotations directory.

### Annotation File Format
Each line in the annotations.txt file corresponds to an annotation. Each line contains 10+ columns, separated by spaces. The definition of these columns are:

    1   Track ID. All rows with the same ID belong to the same path.
    2   xmin. The top left x-coordinate of the bounding box.
    3   ymin. The top left y-coordinate of the bounding box.
    4   xmax. The bottom right x-coordinate of the bounding box.
    5   ymax. The bottom right y-coordinate of the bounding box.
    6   frame. The frame that this annotation represents.
    7   lost. If 1, the annotation is outside of the view screen.
    8   occluded. If 1, the annotation is occluded.
    9   generated. If 1, the annotation was automatically interpolated.
    10  label. The label for this annotation, enclosed in quotation marks.
    
### Homography
In order to project the trajectories into real-world coordinates, you should use the values provided in [`estimated_scales.yaml`](./estimated_scales.yaml).
* Note: The certainty values, are supposed to recall if the scale is reliable or not. Some of the scales are estimated using google maps, and some are just a rational guess!

### Load Dataset with Toolkit
In order to the load the datasets, we provided the [`loader_sdd.py`](../../toolkit/loaders/loader_sdd.py)

```python
import os, yaml
from toolkit.loaders.loader_sdd import load_sdd, load_sdd_dir

scene_name = 'hyang'
scene_video_id = 'video6'
# fixme: replace OPENTRAJ_ROOT with the address to root folder of OpenTraj
sdd_root = os.path.join(OPENTRAJ_ROOT, 'datasets', 'SDD')
annot_file = os.path.join(sdd_root, scene_name, scene_video_id, 'annotations.txt')

# load the homography values
with open(os.path.join(sdd_root, 'estimated_scales.yaml'), 'r') as hf:
    scales_yaml_content = yaml.load(hf, Loader=yaml.FullLoader)
scale = scales_yaml_content[scene_name][scene_video_id]['scale']

traj_dataset = load_sdd(annot_file, scale=scale, scene_id=scene_name + '-' + scene_video_id,
                        drop_lost_frames=False, use_kalman=False) 
```

## License
The datasets is published under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](http://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact [Authors](mailto:amirabs@stanford.edu) for further options.

### Citation
```
@inproceedings{robicquet2016learning,
  title={Learning social etiquette: Human trajectory understanding in crowded scenes},
  author={Robicquet, Alexandre and Sadeghian, Amir and Alahi, Alexandre and Savarese, Silvio},
  booktitle={European conference on computer vision},
  pages={549--565},
  year={2016},
  organization={Springer}
}
```
