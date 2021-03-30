import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objs as go
from toolkit.benchmarking.load_all_datasets import get_datasets, get_trajlets, all_dataset_names
from toolkit.benchmarking.indicators.motion_properties import speed_avg_of_trajs, acceleration_of_tarjs
from matplotlib import cm

if __name__ == '__main__':
    opentraj_root = sys.argv[1]
    datasets = get_datasets(opentraj_root, ['KITTI'])
    trajlet_sets = get_trajlets(opentraj_root,
                                # all_dataset_names[:1]
                                [
                                 # 'ETH-Univ',
                                 # 'ETH-Hotel',
                                 # 'UCY-Zara',
                                 # 'UCY-Univ',
                                 # 'SDD-coupa',
                                 # 'SDD-bookstore',
                                 # 'SDD-deathCircle',
                                 # 'WildTrack',
                                 # 'KITTI'
                                 'LCas-Minerva',
                                ]
                                )

    colormap = cm.get_cmap('Reds', 10)

    for ds_name, trajlet_set in trajlet_sets.items():
        fig = go.Figure()

        speed_avg = speed_avg_of_trajs(trajlet_set)
        acc_avg, acc_max = acceleration_of_tarjs(trajlet_set)
        print(speed_avg)
        # speed_avg = np.digitize(speed_avg, bins=[0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.4, 1.6, 1.8])
        # speed_avg = (speed_avg - min(speed_avg)) / (max(speed_avg) - min(speed_avg))

        # speed_avg_rep = np.concatenate([[x] * trajlet_set.shape[1] for x in speed_avg])
        # trajlet_index = np.concatenate([np.stack([ii] * trajlet_set.shape[1]) for ii in range(trajlet_set.shape[0])])

        # data_with_trajlet_index = np.concatenate([trajlet_index[:, None],
        #                                           trajlet_set.reshape((-1, trajlet_set.shape[2])),
        #                                           speed_avg_rep[:, None]], axis=1)
        # df = pd.DataFrame(data_with_trajlet_index, columns=["trj_index", "pos_x", "pos_y",
        #                                                     "vel_x", "vel_y", "t", "speed_avg"])

        # fig = px.line(df, x="pos_x", y="pos_y", color="speed_avg",
        #               line_group="trj_index", hover_name="trj_index", title=ds_name)

        for ii, trajlet in enumerate(trajlet_set):
            # plt.plot(trajlet[:, 0], trajlet[:, 1])

            line_color = colormap((speed_avg[ii] - min(speed_avg)) / (max(speed_avg) - min(speed_avg)))
            fig.add_trace(go.Scatter(x=trajlet[:, 0], y=trajlet[:, 1],
                                     mode='lines+markers',
                                     name=f'speed={speed_avg[ii]}',
                                     marker=dict(
                                         color=f'rgb({line_color[0]}, {line_color[1]}, {line_color[2]})',
                                         opacity=0.3),
                                     line=dict(color='royalblue')
                                     ))

        fig.update_layout(title=ds_name, xaxis_title="X", yaxis_title="Y")
        fig.show()
        # plt.show()
        # break

