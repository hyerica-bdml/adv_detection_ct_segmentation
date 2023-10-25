################################################################
################################################################
################################################################
# packages

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--attack_name', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--eps', type=float)
parser.add_argument('--niters', default=None, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--random_seed', default=0, type=int)
parser.add_argument('--organs', default=14)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
import warnings
warnings.filterwarnings('ignore')

# math
import numpy as np

# deep learning
import torch

# visualziation
from tqdm import tqdm
# import matplotlib.pyplot as plt
# %matplotlib inline
# import IPython.display as ipd

# medical domain
import monai

# my package
from models.unet import UNet
from utils.dataloader_ct import Dataloader
from utils.unet_utils import test_unet
from adversarial_attacks.attack_unet import (FGSM, BIM, SMIA, attack_unet_model)
from utils.dataloader_ct import CustomDataset
from send_message.task_watcher_native import send_to_mybot

# io
import pickle
from glob import glob

################################################################
################################################################
################################################################
# constants
ATTACK_NAME = args.attack_name
EPS = args.eps
NITERS = args.niters

BATCH_SIZE = args.batch_size
RANDOM_SEED = args.random_seed
if type(args.organs)==int:
    ORGANS = list(np.arange(args.organs))
else:
    ORGANS = list(args.organs)

PATH_CKPT = f'/mnt/mydata/whlee/myckpts/unet_2023/my-unet-seed-{RANDOM_SEED}'
PATH_MARK = f"_{ATTACK_NAME}_{EPS}_{NITERS}"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print('ATTACK_NAME', ATTACK_NAME)
print('args.gpu', args.gpu)
print("EPS", EPS)
print('NITERS', NITERS)
print('BATCH_SIZE', BATCH_SIZE)
print('RANDOM_SEED', RANDOM_SEED)
print('ORGANS', ORGANS)
print('PATH_CKPT', PATH_CKPT)
print('DEVICE', DEVICE)
print("PATH_MARK", PATH_MARK)

################################################################
################################################################
################################################################
# data

dataloader = Dataloader(random_seed=RANDOM_SEED, batch_size=BATCH_SIZE, organs=ORGANS)
trainloader, testloader = dataloader.get_dataloader()

################################################################
################################################################
################################################################
# model

unet = UNet(n_classes=len(ORGANS))
unet.to(DEVICE)
unet.load_state_dict(torch.load(PATH_CKPT))

################################################################
################################################################
################################################################
# attack

criterion = monai.losses.dice.DiceCELoss(
    include_background=False,
    lambda_dice=.5,
    lambda_ce=.5
)

mykwargs = {
    "model":unet,
    "epsilon":EPS,
    "loss_fn":criterion,
    "organs":ORGANS,
    "device":DEVICE
}

if ATTACK_NAME=='fgsm':
    myattack = FGSM(**mykwargs)
elif ATTACK_NAME=='bim':
    myattack = BIM(**mykwargs)
elif ATTACK_NAME=='smia':
    myattack = SMIA(**mykwargs)

attackkwargs = {
    "epsilons":EPS,
    'niters':NITERS
}

X_train_pert_list, train_perturb_list = attack_unet_model(
    dataloader.X_train[:], dataloader.y_train[:],
    unet,
    myattack,
    DEVICE,
    attackkwargs
)

with open("./pkls/X_train_pert_list"+PATH_MARK, 'wb') as f:
    pickle.dump(X_train_pert_list, f)
with open("./pkls/train_perturb_list"+PATH_MARK, 'wb') as f:
    pickle.dump(train_perturb_list, f)

X_test_pert_list, test_perturb_list = attack_unet_model(
    dataloader.X_test[:], dataloader.y_test[:],
    unet,
    myattack,
    DEVICE,
    attackkwargs
)

with open('./pkls/X_test_pert_list'+PATH_MARK, 'wb') as f:
    pickle.dump(X_test_pert_list, f)
with open('./pkls/test_perturb_list'+PATH_MARK, 'wb') as f:
    pickle.dump(test_perturb_list, f)

train_pertloader = CustomDataset(X_train_pert_list, dataloader.y_train)
train_pertloader = torch.utils.data.DataLoader(train_pertloader, batch_size=BATCH_SIZE, shuffle=False)
train_pert_dice_list = test_unet(unet, train_pertloader, ORGANS, DEVICE)
print('attack result on trainset', np.mean(train_pert_dice_list))

with open('./pkls/train_pert_dice_list'+PATH_MARK, 'wb') as f:
    pickle.dump(train_pert_dice_list, f)

test_pertloader = CustomDataset(X_test_pert_list, dataloader.y_test)
test_pertloader = torch.utils.data.DataLoader(test_pertloader, batch_size=BATCH_SIZE, shuffle=False)
test_pert_dice_list = test_unet(unet, test_pertloader, ORGANS, DEVICE)
print('attack result on testset', np.mean(test_pert_dice_list))

with open('./pkls/test_pert_dice_list'+PATH_MARK, 'wb') as f:
    pickle.dump(test_pert_dice_list, f)

n_files = len(glob('./pkls/*'))
n_attacks = n_files//6
msg = f"{n_attacks}-th attacks have been done."
print(msg)
send_to_mybot(msg)
