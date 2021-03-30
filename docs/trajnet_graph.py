# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from adjustText import adjust_text
import matplotlib.dates as mdates
# import seaborn as sns
# matplotlib.use('TkAgg')

# Read the table
table_file = '/home/cyrus/Dropbox/OpenTraj-paper/presentation/trajnet-leaders.xls'
data = pd.read_excel(table_file, header=0)
ADEs = data["ADE"]
FDEs = data["FDE"]
date_objs = []
for date in data["Date"]:
    year_month = date.split('-')
    date_obj = datetime.datetime(year=int(year_month[0]), month=int(year_month[1]), day=1)
    date_objs.append(date_obj)
dates = matplotlib.dates.date2num(date_objs)

# Plots
fig, ax = plt.subplots(nrows=1, ncols=1)

# matplotlib.pyplot.plot_date(dates, FDEs, 'bx', label="FDE")
# name_texts = [plt.text(dates[i], FDEs[i], pred_name, horizontalalignment='center')
#               for i, pred_name in enumerate(data["Predictor"])]

matplotlib.pyplot.plot_date(dates, ADEs, 'bo',
                            label="Average Displacement Error")
name_texts = [plt.text(dates[i], ADEs[i], pred_name,
                       color="magenta" if "DCNet" in pred_name else "black")
              for i, pred_name in enumerate(data["Predictor"])]
plt.ylim([0.21, 0.53])
arrows = adjust_text(name_texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.grid(axis='both', linestyle='--')
plt.ylabel("Prediction Error", fontweight='bold')


plt.xlim([min(dates)-500, max(dates) + 300])
plt.xticks([datetime.datetime(1995, 1, 1),
            datetime.datetime(2016, 1, 1),
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2019, 1, 1),
            datetime.datetime(2020, 1, 1),
            datetime.datetime(2021, 1, 1)], rotation=45)
myFmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(myFmt)
plt.xlabel("Origin of the Idea", labelpad=-15, fontweight='bold')

ax.set_facecolor((0.90, 0.97, 0.82))
plt.legend(loc="lower left")
plt.title("Trajnet Benchmark (World Plane Human-Human Dataset)")
plt.show()
