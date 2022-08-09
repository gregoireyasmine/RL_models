import sys
from os.path import expanduser, exists
from sys import argv
from os import mkdir


sys.path.append(expanduser('~/repos/aeon_mecha_de'))



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

n = int(argv[1])

if exists("video_metadata.csv"):
    df = pd.read_csv("video_metadata.csv")
else:
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

    valid_id_file = expanduser('/nfs/winstor/delab/conf/valid_ids.csv')
    vdf = pd.read_csv(valid_id_file)
    valid_ids = list(vdf.id.values[vdf.real.values == 1])

    df.id = df.id.apply(helpers.fixID, valid_ids=valid_ids)
    helpers.mergeSocial(df)
    helpers.merge(df)


    df = df[~df.id.str.contains(';')]  # solo sessions
    df.to_csv('video_metadata.csv')

    print(df)

<<<<<<< HEAD
videopath = '/nfs/nhome/live/gydegobert/to_annotate/'
=======
dataroot = '/nfs/winstor/delab/data/arena0.1/socialexperiment0/'


>>>>>>> 80313a8fe572d5b6be302833d2a98f85e65a7d74
def exportVideos(i, limit=1):
    for k, session in enumerate(df.itertuples()):
        if k == i:
            print('exporting')
     #      " print(f'Exporting video for {helpers.getSessionID(session)}')"
            os.mkdir(videopath + str(i))
            helpers.exportVideo(dataroot, session, datadir=videopath + str(i), force=False)


exportVideos(n)
