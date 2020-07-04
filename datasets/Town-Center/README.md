# Town Center Dataset

[Ref](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets)

## Format

Tracker Output file format
The ground truth and tracking output is provided in the '.top' file format. This consists of rows in comma-seperated variable (CSV) format:

personNumber, frameNumber, headValid, bodyValid, headLeft, headTop, headRight, headBottom, bodyLeft, bodyTop, bodyRight, bodyBottom

- personNumber - A unique identifier for the individual person
- frameNumber - The frame number (counted from 0)
- headValid - 1 if the head region is valid, 0 otherwise
- bodyValid - 1 if the body region is valid, 0 otherwise
- headLeft,headTop,headRight,headBottom - The head bounding box in pixels
- bodyLeft,bodyTop,bodyRight,bodyBottom - The body bounding box in pixels
