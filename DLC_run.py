from os import listdir
import deeplabcut
from sys import argv

n = argv[1]

dest = '/nfs/winstor/delab/data/arena0.1/socialexperiment0_preprocessed/'
config_path = '/nfs/nhome/live/gydegobert/dlc_out_for_gregoire/dlc_project/config.yaml'
video_path = '/nfs/nhome/live/gydegobert/to_annotate/' + str(n) + '/' \
             + listdir('/nfs/nhome/live/gydegobert/to_annotate/' + str(n))[0]


deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True, destfolder=dest)

