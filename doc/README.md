# How to use `readme_builder.py`
```shell script
cd [opentraj_root]/doc
python readme_buider.py --download-tables --build
python readme_buider.py --confirm  
```

**Warning [1]**: this line will overwrite the current data in `README.md` tables.

**Warning [2]**: The builder script looks for `<!--begin(table_[xxx])-->` and `<!--end(table_[xxx])-->` keywords in `README.md`. Then be careful to not change that lines.

#### Setup
before running builder make sure that you have installed the following packages:
```shell script
pip install xlrd numpy pandas re2
```

**Warning [3]**: The tables (`.xls` documents) are used **only** for building the readme. Any changes to those files will be overwritten by running the builder.
You should modify the table files **only** from here (on ethercalc.org):

- [opentraj-public-datasets.xls](https://ethercalc.org/5xdmtogai5l8)
- [opentraj-benchmarks.xls](https://ethercalc.org/bzn1f11s4w2b)

