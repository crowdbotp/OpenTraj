# ATC pedestrian tracking dataset
[ATC dataset](https://irc.atr.jp/crest2010_HRI/ATC_dataset/) is collected using a tracking environment in the "ATC" shopping center in Osaka, Japan. 
This is part of project on enabling mobile social robots to work in public spaces.
The system consists of multiple 3D range sensors, covering an area of about 900 m2. 
The data provided here was collected between October 24, 2012 and November 29, 2013.
In general the data collection was done every week on Wednesday and Sunday, from morning until evening (9:40-20:20).

<p align='center'>
  <img src='./reference.png' width=480 \>
</p>

## Annotations
The data is provided as CSV files, one file for each day (file names are in the format atc-YYYYMMDD.csv).

Each row in a CSV file corresponds to a single tracked person at a single instant, and it contains the following fields:

```matlab
time [ms] (unixtime + milliseconds/1000),
person id,
position x [mm],
position y [mm],
position z (height) [mm],
velocity [mm/s],
angle of motion [rad],
facing angle [rad]
```

## Load Dataset with Toolkit
The loader for this dataset is not implemented :(

## License
> The datasets are free to use for research purposes only. In case you use the datasets in your work please be sure to cite the reference paper below.

## Citation
```
@article{brvsvcic2013person,
  title={Person tracking in large public spaces using 3-D range sensors},
  author={Br{\v{s}}{\v{c}}i{\'c}, Dra{\v{z}}en and Kanda, Takayuki and Ikeda, Tetsushi and Miyashita, Takahiro},
  journal={IEEE Transactions on Human-Machine Systems},
  volume={43},
  number={6},
  pages={522--534},
  year={2013},
  publisher={IEEE}
}
```
For any questions concerning the datasets please contact: konda@atr.jp

