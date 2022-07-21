import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib as mpl
COLORS = mpl.colors.CSS4_COLORS

# HPC path (Gregoire data)
# data_path = '/nfs/nhome/live/gydegobert/solo_sims'


# Local path (Jeff's CSV)
sims = pd.read_csv('solo_sim.csv')
sims['gamma'] = [(1 + (k % 9))/10 for k in range(len(sims))]
sims['beta'] = [(1 + (k // 9) % 9)/10 for k in range(len(sims))]
sims['alpha'] = [(1 + (k // 81) % 99)/100 for k in range(len(sims))]

fig, ax = plt.subplots(3)


def scat(axes, alpha, beta, gamma, color, label):
    sims_to_plot = sims.loc[sims['gamma'] == alpha]
    sims_to_plot = sims_to_plot.loc[sims['beta'] == beta]
    sims_to_plot = sims_to_plot.loc[sims['alpha'] == gamma]
    for occ in sims_to_plot['occ']:
        occ = ast.literal_eval(occ)
        axes[0].scatter(occ[0], occ[2], color=color, label=label)
        axes[1].scatter(occ[2], occ[3], color=color, label=label)
        axes[2].scatter(occ[0], occ[3], color=color, label=label)


for i, alpha in enumerate(np.arange(0.01, 1, 0.01)):
    color = COLORS[list(COLORS.keys())[i]]
    beta = 0.2
    gamma = 0.2
    scat(ax, alpha, beta, gamma, color=color, label='alpha = ' + str(alpha))




plt.show()


"""
for alpha in np.arange(0.01, 1, 0.01):
    for beta in np.arange(0.1, 1, 0.1):
        for gamma in np.arange(0.1, 1, 0.1):
            
"""