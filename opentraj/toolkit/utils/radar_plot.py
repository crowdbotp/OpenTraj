# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class BenchmarkPolygon:
    def __init__(self, metric_names: list):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        size_out = 10
        self. size_in = 8
        self.ax.set_xlim([-size_out, size_out])
        self.ax.set_ylim([-size_out, size_out])
        self.metric_names = metric_names

        N = len(metric_names)
        self.node_angles = np.array([i * 2 * np.pi / N for i in range(N)]) + np.pi / 2
        self.node_coords = np.array([np.cos(self.node_angles), np.sin(self.node_angles)]).T * self.size_in

        for ii, node_i in enumerate(self.node_coords):
            node_next = self.node_coords[((ii + 1) % N)]
            self.ax.plot([node_i[0], node_next[0]], [node_i[1], node_next[1]], color='blue')
            self.ax.plot([0, node_i[0]], [0, node_i[1]], 'b--', linewidth=1)
            if node_i[0] < -0.1:        ha = 'right'
            elif node_i[0] > 0.1:       ha = 'left'
            else:                       ha = 'center'
            if node_i[1] < -0.1:        va = 'top'
            elif node_i[1] > 0.1:       va = 'bottom'
            else:                       va = 'center'
            self.ax.text(node_i[0], node_i[1], metric_names[ii], horizontalalignment=ha, verticalalignment=va)

    def set_data(self, data: pd.DataFrame):
        for jj, metric in enumerate(self.metric_names):
            if metric not in data:
                raise ValueError("incomplete input data, [%s] is missing" % metric)

        patches = []
        for ii in range(len(data)):
            item_nodes = []
            for jj, metric in enumerate(self.metric_names):
                value_i_j = data[metric].iloc[ii]
                item_nodes.append(self.node_coords[jj] * value_i_j)

            random_color = "#"+''.join([random.choice('56789ABCDEF') for _ in range(6)])

            poly = Polygon(item_nodes, True, alpha=0.8, fill=False, edgecolor=random_color, linewidth=3)
            patches.append(poly)
            self.ax.add_patch(poly)


if __name__ == "__main__":
    metrics = ['predictability',
               'linearity',
               'density',
               'multi-modality',
               'scene-complexity',
               'collision-energy']

    df = pd.DataFrame({metrics[0]: [0.7, 0.62, 0.5],
                       metrics[1]: [0.7, 0.8, 0.6],
                       metrics[2]: [0.5, 0.45, 0.6],
                       metrics[3]: [0.7, 0.4, 0.45],
                       metrics[4]: [0.4, 0.5, 0.6],
                       metrics[5]: [0.7, 0.2, 0.5],
                       })

    bp = BenchmarkPolygon(metrics)
    bp.set_data(df)
    plt.show()
