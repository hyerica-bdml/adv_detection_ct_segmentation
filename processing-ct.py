import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--organs', default=[None], type=list)
args = parser.parse_args()

import warnings
warnings.filterwarnings('ignore')

# math
import numpy as np

# my packages
from utils.dataloader_ct import Dataloader


#########################################################
#########################################################
#########################################################
# CONSTANT
RANDOM_SEED = args.seed
BATCH_SIZE = args.batch_size

if None in args.organs:
    ORGANS = list(np.arange(14))
else:
    ORGANS = args.organs

print('BATCH_SIZE', BATCH_SIZE)
print('RANDOM_SEED', RANDOM_SEED)
print('ORGANS', ORGANS)
#########################################################
#########################################################
#########################################################
# data processing
dataloader = Dataloader(random_seed=RANDOM_SEED, batch_size=BATCH_SIZE, organs=ORGANS)
