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

content = in_file.readlines()
for i, row in enumerate(content):
    row = row.split(' ')
    while '' in row: row.remove('')
    if len(row) < 5: continue
    id = int(float(row[1]))
    ts = int(float(row[0]))
    U = float(row[2])
    V = float(row[4])
    px, py = project(H_iw, np.array([U, V]))

    out_file.write('%e   %e   %e   %e   %e   %e   %e   %e\n' % (ts, id, px, 0, py, 0, 0, 0))
