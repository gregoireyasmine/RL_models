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
    inv = np.linalg.inv(cov)
    return -0.5 * (log(abs(det)) + np.dot(np.dot((x - mean), inv), (x-mean)) + 3*log(2*pi))


dataocc = np.load('dlcoccupancydata.npy', allow_pickle=True).item()

dataocc = np.array([dataocc[k] for k in dataocc.keys()])

dataocc = np.mean(dataocc, axis=0)

parameters = np.array([[[(alpha, beta, gamma) for gamma in [k/10 for k in range(1, 10)]] for beta in [k/10 for k in range(1, 10)]] for alpha in [k/100 for k in range(1, 100)]])
likelihood = np.zeros(np.shape(parameters)[:-1])
errors = 0
total = 0
for a, alpha in enumerate([k/100 for k in range(1, 100)]):
    for b, beta in enumerate([k/10 for k in range(1, 10)]):
        for g, gamma in enumerate([k/10 for k in range(1, 10)]):
            sims_to_plot = sims.loc[sims['alpha'] == alpha]
            sims_to_plot = sims_to_plot.loc[sims['beta'] == beta]
            sims_to_plot = sims_to_plot.loc[sims['gamma'] == gamma]
            occ = sims_to_plot['occ']
            occ = np.array([[ast.literal_eval(oc)[k] for k in [0, 2, 3]] for oc in occ])
            func = gaussian3d_fit(occ)
            try:
                lklh = func(dataocc)
            except Exception as error:
                print('error: ', error)
                errors += 1
                lklh = None
            total += 1
            likelihood[a, b, g] = lklh
print('max_likelihood: ', np.nanmax(likelihood))
print(parameters[np.where(likelihood == np.nanmax(likelihood))])
print('error rate = ', errors / total)
fig = plt.figure()
ax = plt.axes(projection='3d')
alpha = [k/100 for k in range(1, 100)]
beta = [k/10 for k in range(1, 10)]
gamma = [k/10 for k in range(1, 10)]
X, Y = np.meshgrid(alpha, beta)
Z = np.zeros(np.shape(X))
for a in range(0, 99):
    for b in range(0, 9):
        g = 4
        try:
            Z[b, a] = likelihood[a, b, g]
        except IndexError:
            print(np.shape(Z))
            print(a, b)
            print(np.shape(likelihood))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.savefig('3Dcurve')

fig, ax = plt.subplots(1, 3, figsize=(15, 6))
plots = [["alpha", "beta"], ["alpha", "gamma"], ["beta", "gamma"]]
params = {"alpha": alpha, "beta": beta, "gamma": gamma}
for k in range(3):
    param1 = plots[k][0]
    param2 = plots[k][1]
    param1_val = params[param1]
    param2_val = params[param2]
    print(param1, param2)
    X, Y = np.meshgrid(param2_val, param1_val)
    Z = np.zeros((np.shape(X)))
    first = 0
    for p1 in range(len(param1_val)):
        for p2 in range(len(param2_val)):
            try:
                Z[p1, p2] = np.nanmax(likelihood, axis=2-k)[p1, p2]
            except IndexError:
                if first == 0:
                    print(np.nanmax(likelihood, axis=2-k))
                    print(np.shape(np.nanmax(likelihood, axis=2-k)))
                    print(p1, p2)
                    print(Z)
                    first = 1
                else:
                    print(p1, p2)
    c = ax[k].imshow(Z, cmap='RdBu', vmin=np.min(Z), vmax=np.max(Z),
                     extent=[X.min(), X.max(), Y.min(), Y.max()],
                     interpolation='nearest', origin='lower')
    ax[k].set_xlabel(param1)
    ax[k].set_ylabel(param2)
    fig.colorbar(c, ax=ax[k])
fig.tight_layout()
plt.savefig('parameters likelihood')









"""
for alpha in np.arange(0.01, 1, 0.01):
    for beta in np.arange(0.1, 1, 0.1):
        for gamma in np.arange(0.1, 1, 0.1):
"""