import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib as mpl
from math import log, pi
COLORS = mpl.colors.CSS4_COLORS
from mpl_toolkits import mplot3d

# HPC path (Gregoire data)
data_path = '/nfs/winstor/delab/data/'
sims = pd.read_csv(data_path + 'solo_sim.csv')


# Local path (Jeff's CSV)

# sims = pd.read_csv('data/solo_sim.csv')
sims['gamma'] = [(1 + (k % 9))/10 for k in range(len(sims))]
sims['beta'] = [(1 + (k // 9) % 9)/10 for k in range(len(sims))]
sims['alpha'] = [(1 + (k // 81) % 99)/100 for k in range(len(sims))]

fig, ax = plt.subplots(3)


def scat(axes, alpha, beta, gamma, color, label):
    sims_to_plot = sims.loc[sims['alpha'] == alpha]
    sims_to_plot = sims_to_plot.loc[sims['beta'] == beta]
    sims_to_plot = sims_to_plot.loc[sims['gamma'] == gamma]
    for occ in sims_to_plot['occ']:
        occ = ast.literal_eval(occ)
        axes[0].scatter(occ[0], occ[2], color=color, label=label)
        axes[1].scatter(occ[2], occ[3], color=color, label=label)
        axes[2].scatter(occ[0], occ[3], color=color, label=label)

'''
for i, alpha in enumerate(np.arange(0.01, 1, 0.01)):
    color = COLORS[list(COLORS.keys())[i]]
    beta = 0.2
    gamma = 0.2
    scat(ax, alpha, beta, gamma, color=color, label='alpha = ' + str(alpha))
'''

# plt.show()


def gaussian3d_fit(occ):
    mean = np.mean(occ, axis=0)
    cov = np.cov(occ, rowvar=0)
    return lambda x: log_likelihood(x, mean, cov)


def log_likelihood(x, mean, cov):
    det = np.linalg.det(cov)
    print(det)
    inv = np.linalg.inv(cov)
    return -0.5 * (log(det) + np.dot(np.dot((x - mean), inv), (x-mean)) + 3*log(2*pi))


dataocc = np.load('dlcoccupancydata.npy', allow_pickle=True).item()

dataocc = np.array([dataocc[k] for k in dataocc.keys()])

dataocc = np.mean(dataocc, axis=0)

parameters = np.array([[[(alpha, beta, gamma) for gamma in np.arange(0.1, 1, 0.1)] for beta in np.arange(0.1, 1, 0.1)] for alpha in np.arange(0.01, 1, 0.01)])
likelihood = np.zeros(np.shape(parameters))
errors = 0
total = 0
for a, alpha in enumerate(np.arange(0.01, 1, 0.01)):
    for b, beta in enumerate(np.arange(0.1, 1, 0.1)):
        for g, gamma in enumerate(np.arange(0.1, 1, 0.1)):
            sims_to_plot = sims.loc[sims['alpha'] == alpha]
            sims_to_plot = sims_to_plot.loc[sims['beta'] == beta]
            sims_to_plot = sims_to_plot.loc[sims['gamma'] == gamma]
            occ = sims_to_plot['occ']
            occ = np.array([ast.literal_eval(oc) for oc in occ])
            func = gaussian3d_fit(occ)
            try:
                lklh = func(dataocc)
            except Exception as error:
                print(error)
                errors += 1
                lklh = None
            total += 1
            likelihood[a, b, g] = lklh
print(parameters[np.where(likelihood == np.max(likelihood))])
print('error rate = ', errors / total)
fig = plt.figure()
ax = plt.axes(projection='3d')
X = []
Y = []
Z = []
for a in range(np.shape(parameters)[0]):
    for b in range(np.shape(parameters)[1]):
        g = 0
        X.append(parameters[a, b, g][0])
        Y.append(parameters[a, b, g][1])
        Z.append(likelihood[a, b, g])
print(X, Y, Z)
ax.contour3D(X, Y, Z, 50, cmap='binary')
plt.savefig('3Dcurve')



"""
for alpha in np.arange(0.01, 1, 0.01):
    for beta in np.arange(0.1, 1, 0.1):
        for gamma in np.arange(0.1, 1, 0.1):
"""