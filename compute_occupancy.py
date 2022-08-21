import numpy as np
from os import listdir

sessions = listdir('./dlcoccdata/')
occupancy = {}
for session in sessions:
    data = np.load('./dlcoccdata/'+session, allow_pickle=True)
    data = data[:, 1]
    occ = [np.sum(data == k)/len(data) for k in range(2)]
    occupancy[session] = occ
np.save('dlcoccupancydata', occupancy)
