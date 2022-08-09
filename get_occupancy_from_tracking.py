from os import listdir
import pandas as pd
from math import sqrt
## This is meant to be ran on HPC

data_path = '/nfs/winstor/delab/data/arena0.1/socialexperiment0_preprocessed/'
ROI_distance_quotient = 0.17
sessions = listdir(data_path)
df = pd.DataFrame(columns = ['frame', 'patch_occupied'])

for sess in sessions:
    filename = data_path + sess + listdir(data_path + sess)[0]
    sessdf = pd.read_csv(filename)

    rwx, rwy, lwx, lwy = None, None, None, None
    frame = 0
    while sum([type(w) == float for w in (rwx, rwy, lwx, lwy)]) != 4:
        rwx = sessdf.iloc[frame]['single_right_wheel_x']
        rwy = sessdf.iloc[frame]['single_right_wheel_y']
        lwx = sessdf.iloc[frame]['single_left_wheel_x']
        lwy = sessdf.iloc[frame]['single_left_wheel_y']
        frame += 1
    distance_between_patches = sqrt((rwx - lwx)**2 + (rwy - lwy)**2)
    patch_radius = ROI_distance_quotient * distance_between_patches
    for i, frame in sessdf.itertuples():
        x = frame.ind1_nose_x 
        y = frame.ind1_nose_y
        if x and y:
            patch_occupied = is_on_patch(rwx, rwy, lwx, lwy, x, y, patch_radius)
        else: 
            patch_occupied = None
        df.append({'frame': i, 'patch_occupied': patch_occupied})
                   
            
def belong_to_circle(ptx, pty, cx, cy, radius):
    """ Returns if a point belongs to a circle. """
    return (ptx - cx)**2 + (pty - cy)**2 < radius**2


def is_on_patch(patch1_x, patch1_y, patch2_x, patch2_y, ind_x, ind_y, radius):
    """ Returns i if the individual is on patch i """
    for i, px, py in enumerate([(patch1_x, patch1_y), (patch2_x, patch2_y)]):
        if belong_to_circle(ind_x, ind_y, px, py, radius):
            return i+1
    return 0
        
        

