import sys
from os.path import expanduser

sys.path.append(expanduser('~/repos/aeon_mecha_de'))

dataroot = '/nfs/winstor/delab/data/arena0.1/socialexperiment0_raw/'

import numpy as np
import pandas as pd
import aeon.analyze.patches as patches
import os
import aeon.preprocess.api as api
import matplotlib.pyplot as plt

import aeon.util.helpers
import aeon.util.plotting

helpers = aeon.util.helpers
plotting = aeon.util.plotting

sessdf = api.sessiondata(dataroot)
sessdf = api.sessionduration(sessdf)                    
sessdf = sessdf[~sessdf.id.str.contains('test')]
sessdf = sessdf[~sessdf.id.str.contains('jeff')]
sessdf = sessdf[~sessdf.id.str.contains('OAA')]
sessdf = sessdf[~sessdf.id.str.contains('rew')]
sessdf = sessdf[~sessdf.id.str.contains('Animal')]
sessdf = sessdf[sessdf.loc[:, 'start'] > np.datetime64("2022-01-01")]


sessdf.reset_index(inplace=True, drop=True) 

df = sessdf.copy()

pd.set_option('display.max_rows', 500)

valid_id_file = expanduser('/home/mnt/delab/conf/valid_ids.csv')
vdf = pd.read_csv(valid_id_file)
valid_ids = list(vdf.id.values[vdf.real.values==1])

df.id = df.id.apply(helpers.fixID, valid_ids=valid_ids)
helpers.mergeSocial(df)
helpers.merge(df)


videopath='/home/gregoiredy/to_annotate'


df = df[~df.id.str.contains(';')]  # solo sessions
print(df)


def exportVideos(limit=1):
    done = 0

    for session in df.itertuples():
        print(f'Exporting video for {helpers.getSessionID(session)}')

        helpers.exportVideo(dataroot, session, datadir=videopath, force=False)

        if done >= limit:
            return
        else:
            done += 1


#  exportVideos()
