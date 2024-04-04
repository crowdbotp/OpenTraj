#!/bin/bash
echo "Downloading the zip files from Edinburgh pedestrian dataset."
for day in $(cat days.txt)
do
    curl https://homepages.inf.ed.ac.uk/rbf/FORUMTRACKING/DAYS/$day.zip > $day.zip
done
