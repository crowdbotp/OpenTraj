The annotation data are taken from [milanton.de](http://www.milanton.de/data/)

All annotations were done by manually placing bounding boxes around pedestrians and interpolating their trajectories between key frames. The trajectories were then smoothed to avoid discontinuous or jittery movements. All targets were annotated in all sequences, even in case of total occlusion. Each person entering the field of view acquires a unique ID, i.e. if a person leaves the screen and later reappears again, a new ID is assigned. Please note that bounding boxes are not always perfectly aligned due to articulation, interpolation and mistakes made by the annotator.
The annotations are saved in xml files using the CVML Specification, similar to the ground truth of the CAVIAR dataset.
For each dataset there are two ground truth files. One (complete) contains annotations for all visible targets, the other one (cropped) only contains targets within the predefined tracking area used in our experiments.
Please cite our work if you use the provided ground truth.

- All annotations were done in View001.

