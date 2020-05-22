import numpy as np


def project(Hinv, loc):
    locHomogenous = np.hstack((loc, 1))
    locHomogenous = np.dot(Hinv, locHomogenous)  # to camera frame
    locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
    return locXYZ[:2]


annot_file = './obsmat_px.txt'
annot_file_out = './obsmat.txt'
H_iw = np.loadtxt('./H.txt')

in_file = open(annot_file, 'r')
out_file = open(annot_file_out, 'w')

class Trajectory:
    def __init__(self):
        self.id = -1
        self.pos = []
        self.vel = []
        self.t_list = []

content = in_file.readlines()
trajs = {}
for i, row in enumerate(content):
    row = row.split(' ')
    while '' in row: row.remove('')
    if len(row) < 5: continue
    id = int(float(row[1]))
    if id not in trajs:
        trajs[id] = Trajectory()

    ts = int(float(row[0]))
    U = float(row[2])
    V = float(row[4])
    px, py = project(H_iw, np.array([U, V]))

    trajs[id].id = id
    trajs[id].t_list.append(ts)
    trajs[id].pos.append([px, py])


for id, traj in trajs.items():
    traj.pos = np.array(traj.pos)
    traj.vel = np.zeros_like(traj.pos)
    traj.vel[:-1, :] = np.diff(traj.pos, axis=0) * 2.5
    traj.vel[-1] = traj.vel[-2]


for i, row in enumerate(content):
    row = row.split(' ')
    while '' in row: row.remove('')
    if len(row) < 5: continue
    id = int(float(row[1]))
    ts = int(float(row[0]))

    t_ind = trajs[id].t_list.index(ts)

    px, py = trajs[id].pos[t_ind, :]
    vx, vy = trajs[id].vel[t_ind, :]

    out_file.write('% e   % e   % e   % e   % e   % e   % e   % e\n' % (ts, id, px, 0, py, vx, 0, vy))
