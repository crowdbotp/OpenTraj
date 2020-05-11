## `readme_builder.py`
This script is created to automatize the process of generating the markdown tables in main `README.md`. In fact both main table (public datasets) and the benchmark tables are stored on a web spreadsheet, in the following address:
- [opentraj-public-datasets.xls](https://ethercalc.org/5xdmtogai5l8)
- [opentraj-benchmarks.xls](https://ethercalc.org/bzn1f11s4w2b)

One can modify these tables (everyone with those link has read/write permission). And then run the script to update the README file.

#### Setup
Before running the builder, make sure that you havethe following packages installed:
```shell script
pip install xlrd numpy pandas re2
```

#### How to use it
```shell script
cd [opentraj_root]/doc
python readme_buider.py --download-tables --build  # for creating README_temp.md
python readme_buider.py --confirm       # for overwriting the origianl README.md
```

**Warning [1]**: this line will overwrite the current data in `README.md` tables.

**Warning [2]**: The builder script looks for `<!--begin(table_[xxx])-->` and `<!--end(table_[xxx])-->` keywords in `README.md`. Then be careful to not change that lines.

**Warning [3]**: The tables (`.xls` documents) are used **only** for building the readme. Any changes to those files will be overwritten by running the builder.
You should modify the table files **only** from here (on ethercalc.org):

