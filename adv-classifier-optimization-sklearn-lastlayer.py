import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--attack_name', type=str)
parser.add_argument('--classifier', type=int)
parser.add_argument('--gpu', default=1, type=int)
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
import inspect

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import scipy

from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


###########################################################################
###########################################################################
###########################################################################
# CONSTANT
CLASSIFIER_LIST = [
    RandomForestClassifier, 
    GaussianProcessClassifier, 
    RBF,
    GaussianNB,
    KNeighborsClassifier,
    MLPClassifier,
    SVC,
    DecisionTreeClassifier
]

CLASSIFIER = CLASSIFIER_LIST[int(args.classifier)]
CLF_NAME = CLASSIFIER.__name__

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

print('CLASSIFIER', CLASSIFIER)
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

###########################################################################
###########################################################################
###########################################################################
# DATA
dataloader = Dataloader(random_seed=RANDOM_SEED, batch_size=BATCH_SIZE, organs=ORGANS)
trainloader, testloader = dataloader.get_dataloader()

###########################################################################
###########################################################################
###########################################################################
# MODEL
unet = UNet(n_classes=len(ORGANS))
unet.to(DEVICE)
unet.load_state_dict(torch.load(PATH_CKPT))

###########################################################################
###########################################################################
###########################################################################
# ATTACK
criterion = monai.losses.dice.DiceCELoss(
    include_background=False,
    lambda_dice=.5,
    lambda_ce=.5
)

with open('./pkls/X_train_pert_list'+PATH_MARK, 'rb') as f:
    X_train_pert_list = pickle.load(f)
    # print('X_train_pert_list loaded', type(X_train_pert_list))
# with open("./pkls/train_perturb_list"+PATH_MARK, 'wb') as f:
#     pickle.dump(train_perturb_list, f)
with open('./pkls/train_perturb_list'+PATH_MARK, 'rb') as f:
    train_perturb_list = pickle.load(f)
    # print('train_perturb_list loaded', type(train_perturb_list))

with open('./pkls/X_test_pert_list'+PATH_MARK, 'rb') as f:
    X_test_pert_list = pickle.load(f)
    # print('X_test_pert_list loaded', type(X_test_pert_list))
# with open('./pkls/test_perturb_list'+PATH_MARK, 'wb') as f:
#     pickle.dump(test_perturb_list, f)
with open('./pkls/test_perturb_list'+PATH_MARK, 'rb') as f:
    test_perturb_list = pickle.load(f)
    # print('test_perturb_list loaded', type(test_perturb_list))

with open('./pkls/train_pert_dice_list'+PATH_MARK, 'rb') as f:
    train_pert_dice_list = pickle.load(f)
    # print('train_pert_dice_list loaded', type(train_pert_dice_list))

with open('./pkls/test_pert_dice_list'+PATH_MARK, 'rb') as f:
    test_pert_dice_list = pickle.load(f)
    # print('test_pert_dice_list loaded', type(test_pert_dice_list))

###########################################################################
###########################################################################
###########################################################################
# ATTACK DETECTION

def get_hidden_features_unet(X, unet, device):
    '''
    feature from the last layer
    '''
    unet.eval()
    unet.to(device)
    feature_list = list()
    for x in tqdm(X):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = x.to(device)
        if len(x.size()) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        x1 = unet.inc(x)
        x2 = unet.down1(x1)
        x3 = unet.down2(x2)
        x4 = unet.down3(x3)
        x5 = unet.down4(x4)
        x = unet.up1(x5, x4)
        x = unet.up2(x, x3)
        x = unet.up3(x, x2)
        x = unet.up4(x, x1)
        feature = x # last conv
        feature_list.append(feature.detach().cpu().numpy())

    return np.array(feature_list)

train_feature_original = get_hidden_features_unet(dataloader.X_train, unet, DEVICE)
train_feature_pert = get_hidden_features_unet(X_train_pert_list, unet, DEVICE)

def get_pure_features(
    filter,
    feature_original,
    feature_pert,
):
    X_test_filter, y_test_filter = [], []
    for f in tqdm(feature_original):
        X_test_filter.append(f.squeeze()[filter].flatten())
        y_test_filter.append(0)
    for f in tqdm(feature_pert):
        X_test_filter.append(f.squeeze()[filter].flatten())
        y_test_filter.append(1)
    return X_test_filter, y_test_filter

def get_split_features(
    orig_features, 
    pert_features, 
    filter, 
    from_features=True, 
    plot_hist=False,
    random_state=0,
    X_pert_list=None,
    tqdm_off=True,
    verbose=False,
    test=True,
):
'''
    args:
        orig_features: Original features.
        pert_features: Perturbed features.
        filter: (int) Filter number or channel number.
        from_features: (bool) If True, select filter from features. If False, select channel from image.
        plot_hist: (bool) Whether to plot histogram or not.
    '''

    f = "features" if from_features else "imgs"
    if verbose:
        print(f"{f}[{filter}, :, :]")

    if from_features:
        pass
    else:
        # for original image
        X_test = list()
        for _x in dataloader.X_test:
            X_test.append(_x)
        _X_pert_list = list()
        for _x in X_pert_list:
            _X_pert_list.append(torch.from_numpy(_x))
        X_pert_list = _X_pert_list

    # split index
    idx_train, idx_test = train_test_split(
        np.arange(len(orig_features)*2), test_size=.5, random_state=random_state)

    # for training
    X_train_filter, y_train_filter = [], []
    for idx in tqdm(idx_train[idx_train<len(orig_features)], disable=tqdm_off):
        if from_features:
            X_train_filter.append(orig_features[idx].squeeze()[filter].flatten())
        else:
            X_train_filter.append(X_test[idx].detach().cpu().numpy().flatten()) 
        y_train_filter.append(0)
    for idx in tqdm(idx_train[idx_train>=len(orig_features)]-len(orig_features), disable=tqdm_off):
        if from_features:
            X_train_filter.append(pert_features[idx].squeeze()[filter].flatten())
        else:
            X_train_filter.append(X_pert_list[idx].detach().cpu().numpy().flatten())
        y_train_filter.append(1)
    X_train_filter, y_train_filter = np.array(X_train_filter), np.array(y_train_filter)
    randidx = np.random.choice(np.arange(len(X_train_filter)), size=len(X_train_filter), replace=False)
    X_train_filter = X_train_filter[randidx]
    y_train_filter = y_train_filter[randidx]

    # split validation and test
    X_test_filter, y_test_filter = [], []
    for idx in tqdm(idx_test[idx_test<len(orig_features)], disable=tqdm_off):
        if from_features:
            X_test_filter.append(orig_features[idx].squeeze()[filter].flatten())
        else:
            X_test_filter.append(X_test[idx].detach().cpu().numpy().flatten())  
        y_test_filter.append(0)
    for idx in tqdm(idx_test[idx_test>=len(orig_features)]-len(orig_features), disable=tqdm_off):
        if from_features:
            X_test_filter.append(pert_features[idx].squeeze()[filter].flatten())
        else:
            X_test_filter.append(X_pert_list[idx].detach().cpu().numpy().flatten()) 
        y_test_filter.append(1)
    X_test_filter, y_test_filter = np.array(X_test_filter), np.array(y_test_filter)
    
    X_valid_filter, X_test_filter, y_valid_filter, y_test_filter = train_test_split(X_test_filter, y_test_filter, test_size=.5, random_state=random_state)

    if plot_hist:
        size = 228
        a = np.array(X_train_filter[y_train_filter==0][np.random.randint(0,np.sum(y_train_filter==0), size=size)]).flatten()
        b = np.array(X_train_filter[y_train_filter==1][np.random.randint(0,np.sum(y_train_filter==1), size=size)]).flatten()

        plt.figure(figsize=(5,3))
        plt.hist(a, bins=50)
        plt.hist(b, alpha=.5, bins=50)
        # plt.xlim(0, 2)
        plt.title(f"{f}[{filter}, :, :], samples={size}, pvalue={scipy.stats.ttest_ind(a, b).pvalue:.4f}")
        plt.show()

    return X_train_filter, y_train_filter,\
    X_valid_filter, X_test_filter,\
    y_valid_filter, y_test_filter

def optimize_filter(
    CLASSIFIER,
    verbose=False
):
    print("optimize_filter", PATH_MARK, CLF_NAME)
    max_val_acc = -1e10
    max_filter = None
    val_acc_list = []
    max_best_hyperparams = None
    
    for nf in tqdm(range(train_feature_original[0].shape[1])):
        if verbose:
            print(nf, '='*80)
        X_train_filter, y_train_filter, X_valid_filter, X_test_filter, y_valid_filter, y_test_filter = get_split_features(
            train_feature_original,
            train_feature_pert,
            filter=nf,
            from_features=True,
            plot_hist=False,
        )
        
        if 'random_state' in inspect.signature(CLASSIFIER.__init__).parameters:
            clf = CLASSIFIER(random_state=0)
        else:
            clf = CLASSIFIER()

        # validation accuracy
        clf.fit(X_train_filter, y_train_filter)
        pred_valid = clf.predict(X_valid_filter)
        val_acc = np.sum(pred_valid == y_valid_filter) / len(pred_valid)
        val_acc_list.append(val_acc)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            max_filter = nf
    return max_val_acc, max_filter, val_acc_list

max_val_acc, max_filter, val_acc_list = optimize_filter(CLASSIFIER=CLASSIFIER)

with open('./pkls/val_acc_list'+'_last_conv_'+PATH_MARK+'_'+CLF_NAME, 'wb') as f:
    pickle.dump(val_acc_list, f)

msg = f"{PATH_MARK[1:]}, {CLF_NAME}, {args.classifier+1}/{len(CLASSIFIER_LIST)}-th done"
print(msg)
send_to_mybot(msg)