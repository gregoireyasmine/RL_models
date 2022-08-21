from os import listdir
import pandas as pd
from math import sqrt
import numpy as np
from sys import argv
n = argv[1]
## This is meant to be ran on HPC

data_path = '/nfs/nhome/live/gydegobert/repos/RL_models/dlcdata/'
ROI_distance_quotient = 0.17
sessions = listdir(data_path)
dict = {}


def belong_to_circle(ptx, pty, cx, cy, radius):
    """ Returns if a point belongs to a circle. """
    return (ptx - cx) ** 2 + (pty - cy) ** 2 < radius ** 2


def is_on_patch(patch1_x, patch1_y, patch2_x, patch2_y, ind_x, ind_y, radius):
    """ Returns i if the individual is on patch i """
    for i, p in enumerate([(patch1_x, patch1_y), (patch2_x, patch2_y)]):
        px, py = p
        if belong_to_circle(ind_x, ind_y, px, py, radius):
            return i + 1
    return 0


sess = sessions[n]
df = pd.DataFrame(columns=['frame', 'patch_occupied'])
filename = data_path + sess
try:
    sessdf = pd.read_csv(filename, header=[0, 1, 2, 3], encoding="ISO-8859-1")
    print('analyzing ', sess, ' DLC data')
    cols = []

    for animal in ['704', '705', '706']:
        if animal in sess:
            ind = animal

    for col in sessdf.columns:
        if col[1] == ind:
            column = 'ind1' + '_' + col[2] + '_' + col[3]
        else:
            column = col[1] + '_' + col[2] + '_' + col[3]
        cols.append(column)
    sessdf.columns = cols

    rwx, rwy, lwx, lwy = None, None, None, None
    frame = 0
    while sum([type(w) == np.float64 for w in (rwx, rwy, lwx, lwy)]) != 4:
        rwx = sessdf.iloc[frame]['single_right_wheel_x']
        rwy = sessdf.iloc[frame]['single_right_wheel_y']
        lwx = sessdf.iloc[frame]['single_left_wheel_x']
        lwy = sessdf.iloc[frame]['single_left_wheel_y']
        frame += 1
    distance_between_patches = sqrt((rwx - lwx)**2 + (rwy - lwy)**2)
    patch_radius = ROI_distance_quotient * distance_between_patches

    for i, frame in enumerate(sessdf.itertuples()):
        x = frame.ind1_nose_x
        y = frame.ind1_nose_y
        if x and y:
            patch_occupied = is_on_patch(rwx, rwy, lwx, lwy, x, y, patch_radius)
        else:
            patch_occupied = None
        df = pd.concat([df, pd.DataFrame({'frame': i, 'patch_occupied': patch_occupied}, index=[0])], ignore_index=True)
    np.save('./dlcdata/occupancy_DLC_' + sess + '.npz', df)

except Exception:
    print(Exception)



