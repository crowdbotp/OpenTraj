# Author: Javad Amirian
# Email: amiryan.j@gmail.com

# Run this app with `python app.py` and
import os
import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from toolkit.loaders.loader_crowds import load_crowds
from opentraj.toolkit.test.load_all import all_dataset_names, get_datasets, get_trajlets

# Application
# ===========================================
# see https://plotly.com/python/px-arguments/ for more options
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# ----------------------


# default values
# ----------------------
traj_dataset = pd.DataFrame({})
cur_time = 0
time_horizon = [-4.8, 4.8]
traj_fig = go.Figure()
graph_height = 800
time_horizon_marks = [0.4, 0.8, 1.6, 3.2, 4.8, 9.6]
time_horizon_marks = [-x for x in time_horizon_marks[::-1]] + [0] + time_horizon_marks
cur_time_slider = dcc.Slider(id='cur_time', min=0, max=1000, step=0.1, value=0, updatemode='drag')

# Biuld the layout
# dbc.Spinner()
app.layout = html.Div(children=[
    html.H2('OpenTraj'),
    html.H5('Benchmarking Human Trajectory Prediction Datasets'),
    html.Hr(),
    dbc.Container(['OpenTraj Root: ',
                   dcc.Input(id='opentraj_root', type='text', spellCheck='false',
                             placeholder='Enter path to the root directory of OpenTraj',
                             # value='Enter the path to the root directory of OpenTraj',
                             value='/home/cyrus/workspace2/OpenTraj',
                             inputMode="url", style={'width': 360})],
                  fluid=True, id='opentraj_root_container'),
    dcc.Dropdown(id='select_dataset', placeholder='Select Dataset...',
                 options=[{'label': ds_name, 'value': ds_name} for ds_name in all_dataset_names], multi=False),
    dcc.Graph(id='traj_graph', figure=traj_fig, style={'height': graph_height}),
    cur_time_slider,
    dcc.RangeSlider(id='time_horizon', min=time_horizon_marks[0], max=time_horizon_marks[-1], step=0.1,
                    marks=dict(
                        zip(time_horizon_marks, ['%s%.1fs' % (["", "+"][x > 0.001], x) for x in time_horizon_marks])),
                    value=[-4.8, 4.8], updatemode='drag')
])


@app.callback(Output('cur_time', 'max'),
              Input('opentraj_root', 'value'),
              Input('select_dataset', 'value'))
def load_dataset(opentraj_root, dataset_name):
    if os.path.exists(opentraj_root) \
            and os.path.exists(os.path.join(opentraj_root, 'datasets')) and dataset_name:
        global traj_dataset, traj_fig, cur_time
        cur_time = 0
        traj_datasets = get_datasets(opentraj_root, [dataset_name])
        if dataset_name in traj_datasets:
            traj_dataset = traj_datasets[dataset_name].data
        return np.max(traj_dataset["timestamp"])
    return 0


@app.callback(Output('traj_graph', 'figure'),
              Input('cur_time', 'value'),
              Input('time_horizon', 'value'),
              Input('opentraj_root', 'value'),
              Input('select_dataset', 'value')
              )
def update_graph(cur_time_, time_horizon_, _, __):
    global cur_time, time_horizon
    cur_time, time_horizon = cur_time_, time_horizon_
    fig = go.Figure()
    if len(traj_dataset):
        print("Scatter Trajs...")
        selected_data = traj_dataset[(traj_dataset["timestamp"] >= cur_time + time_horizon[0])
                                     & (traj_dataset["timestamp"] <= cur_time + time_horizon[1])]
        print("len selected data = ", len(selected_data))
        trajs = selected_data.groupby(["scene_id", "agent_id"])
        trajs = [(scene_id, agent_id, tr) for (scene_id, agent_id), tr in trajs]

        cur_poss = []
        for scene_id, agent_id, traj in trajs:
            fig.add_scatter(x=traj["pos_x"], y=traj["pos_y"], name="%s[%d]" % (scene_id, agent_id),
                            mode="lines+markers")
            if len(traj):
                # print(agent_id)
                # print(traj[traj["timestamp"] <= cur_time])
                cur_poss.append(traj[traj["timestamp"] <= cur_time].iloc[-1][["pos_x", "pos_y"]].to_numpy())
        cur_poss = np.stack(cur_poss)
        fig.add_scatter(x=cur_poss[:, 0], y=cur_poss[:, 1], mode="markers", showlegend=False)
        fig['layout']['xaxis'].update(title='', range=[np.min(traj_dataset["pos_x"]), np.max(traj_dataset["pos_x"])],
                                      dtick=2.5, autorange=False)
        fig['layout']['yaxis'].update(title='', range=[np.min(traj_dataset["pos_y"]), np.max(traj_dataset["pos_y"])],
                                      dtick=2.5, autorange=False)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
