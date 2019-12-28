
import os
import numpy as np


rnd_seed = 431

origin = '/home/gerlstefan/data/pipeline/processableDataset/results/tmp/vesselseg_out'

cwd = os.getcwd()
# change directory to origin, and get a list of all files
os.chdir(origin)
all_files = os.listdir()
os.chdir(cwd)

np.random.seed(rnd_seed)
np.random.shuffle(all_files)

selection = all_files[:20]

for item in selection:
    print(item)
