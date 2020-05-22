1. DATASET FORMAT
-----------------

The labels corresponding to each video TRAFx is stored in the file TRAFx_gt.txt, where 'x' is the video number.

Each line in each TRAFx_gt.txt has the following information in the specified oreder:
<frame number>,<#agents in frame>,<bbox top left x>,<bbox top left y>,<bbox bottom right x>,<bbox bottom right y>,<agent1 ID>,...,<bbox top left x>,<bbox top left y>,<bbox bottom right x>,<bbox bottom right y>,<agentN ID>

- bbox: bounding box of the agent.
- All x and y values are in pixels from top left of the corresponding image frame.
- N = #agents in the frame.
- Agents belong to one of the following classes: ped, cycle, scooter, bike, rick, car, bus, truck, others. Agent ID is assigned according to: <class><instance# in class>.
  For example, the first tracked pedestrian in a video has the ID ped0, the 5th tracked rickshaw has the ID rick4 etc.
- #lines in each file = #frames in the corresponding video.



2. DATASET DETAILS
------------------

- The dataset contains 48 Traffic videos so far.
- Each video is processed at 20 fps, and the videos roughly range between between 1000 and 3000 frames.
- The videos are highly heterogeneous, with roughly 5-8 different agent classes represented per frame.
- For each class in each frame, there are multiple instances, totalling between 10 to 20 agents per frame (15-16 agents on average). This makes the videos highly dense as well.