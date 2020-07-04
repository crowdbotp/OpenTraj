This is the Wildtrack dataset.

Annotations:
Frame annotations are in the folder “annotations_positions”.
Corresponding images from 7 cameras are in the folder “Image_subsets”.

In order to recover the 3D location of the target, you have to use the following information.

The “positionID” in .json files are indexed on a 480x1440 grid, X-first, with a 2.5cm spacing. The origin is (-3.0,-9.0)
Therefore,

X = -3.0 + 0.025*ID%480
Y = -9.0 + 0.025*ID/480

The file “rectangles.pom” gives correspondence between rectangle IDs and 2D bounding boxes on each camera.

This files is needed to run algorithms such as 
https://github.com/pierrebaque/DeepOcclusion

It can be reproduced using 
https://github.com/pierrebaque/generatePOMfile

Camera calibrations:
You will find camera calibrations in the folder “calibrations”.
“extrinsic” contains extrinsic camera calibrations
“intrinsic_zero” contains intrinsic calibrations for the images which have been undistorted. These are the ones which should be used for the images provided in this dataset.
“intrinsic_original” contains calibrations for the original images and the original video, which we can provide if you ask.

If you use this dataset, please cite the paper:
“The WILDTRACK Multi-Camera Person Dataset.” T.Chavdarova et al.

https://arxiv.org/pdf/1707.09299.pdf

If you need more unlabelled data, please contact us.