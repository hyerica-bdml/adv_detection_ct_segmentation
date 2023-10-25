import os

import time
from tqdm import tqdm
import numpy as np
import torch
import monai
from glob import glob
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn

import torch.utils.data as data
from pathlib import Path
import random
import sys
import math
import scipy.stats as st

def attack_unet_model(X_test, y_test, model, attack_method, device, kwargs):
    '''
    args:
        X_test: (list) Collection of images to be attacked.
        y_test: (list) Ground truth segment of the images.
        model: Model to be attacked.
        attack_method: Instance of the attack method class.
        device: Device on which the computation will be performed.
        kwargs: (dict) Additional arguments, for example: {"epsilons":1e-3, "niters":5}.
    returns:
        X_pert_list: (list) List of adversarial samples.
        perturb_list: (list) List of perturbations.
    '''

    X_pert_list, perturb_list = [], []

    for idx in tqdm(range(len(X_test))):
        images, labels = X_test[idx].unsqueeze(0).unsqueeze(0).to(device), y_test[idx].unsqueeze(0).unsqueeze(0).to(device)
        perturbed_imgs = attack_method.perturb(images, labels, **kwargs)
        perturbs = images - perturbed_imgs
        X_pert_list.append(perturbed_imgs.squeeze().detach().cpu().numpy())
        perturb_list.append(perturbs.squeeze().detach().cpu().numpy())

    return X_pert_list, perturb_list

def gkern(kernlen=3, nsig=1):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(0,x)
    kern2d = st.norm.pdf(0,x)
    kernel_raw = np.outer(kern1d, kern2d)
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def blur(image, epsilon, stack_kernel):
    min_batch = image.shape[0]
    channels = image.shape[1]
    out_channel = channels
    
    kernel = torch.FloatTensor(stack_kernel).cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)
    data_grad = F.conv2d(image, weight, bias=None, stride=1,padding = 2, dilation=2)
    
    sign_data_grad = data_grad.sign()
    
    perturbed_image = image + epsilon*sign_data_grad
    return data_grad * epsilon


class SMIA(object):
    def __init__(self, model=None, epsilon=None, loss_fn=None, organs=None, device=None):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.device = device
        self.organs = organs
        
    def perturb(self, X, y, epsilons=None, niters=None, a1=1, a2=1):
        model = self.model
        DEVICE = self.device
        ORGANS = self.organs


        if epsilons is not None:
            self.epsilon = epsilons

        use_cuda = torch.cuda.is_available()
        X_pert = X.clone().detach().cuda()
        X_pert.requires_grad = True
        y = y.to(DEVICE, dtype=torch.long)

        for i in range(niters):
            output_perturbed = None
            output_perturbed = model(X_pert.cuda())
            output_perturbed = F.softmax(output_perturbed, dim=1)
            if i == 0:
                dice_y = F.one_hot(y, num_classes=len(ORGANS)).movedim(-1,1).float()
                dice_y = dice_y.squeeze().reshape(output_perturbed.size()) # 모양 맞춰주기
                loss = self.loss_fn(output_perturbed, dice_y)
            else:
                output_perturbed_last = F.softmax(output_perturbed_last, dim=1)
                loss = a1*self.loss_fn(output_perturbed, dice_y) - a2*self.loss_fn(output_perturbed, output_perturbed_last)
            # model.zero_grad()

            loss.backward()
            X_pert_grad = X_pert.grad.detach().sign()
            pert = X_pert_grad * self.epsilon
            # pert = X_pert.grad.detach().sign() * self.epsilon

            kernel = gkern(3,1).astype(np.float32)

            stack_kernel = np.expand_dims(kernel, 0)
            stack_kernel = np.expand_dims(stack_kernel, 1)

            gt1 = X_pert_grad.detach()
            gt1 = blur(gt1, self.epsilon, stack_kernel)
            gt1 = torch.tensor(X_pert + gt1).cuda().clone().detach().requires_grad_(False)
#             gt1.requires_grad = False
            output_perturbed_last = model(gt1)
            X_pert = torch.clamp(torch.tensor(X_pert.cuda() + pert.cuda()),min=0, max=1).clone().detach().requires_grad_(True)
#             X_pert.requires_grad = True
        return X_pert
    
    
class FGSM(object):
    def __init__(self, model=None, epsilon=None, loss_fn=None, organs=None, device=None):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.device = device
        self.organs = organs
        
    def perturb(self, X, y, epsilons=None, niters=None):
        model = self.model
        DEVICE = self.device
        ORGANS = self.organs
        
        if epsilons is not None:
            self.epsilon = epsilons   
        use_cuda = torch.cuda.is_available()   
        X_pert = X.clone().detach().cuda()
        X_pert.requires_grad = True
        y = y.to(DEVICE, dtype=torch.long)
        
        output_perturbed = None
        output_perturbed = model(X_pert.cuda())
        output_perturbed = F.softmax(output_perturbed, dim=1)
    
        dice_y = F.one_hot(y, num_classes=len(ORGANS)).movedim(-1,1).float()
        dice_y = dice_y.squeeze().reshape(output_perturbed.size()) # 모양 안맞을 때 맞춰줌
        
        loss = self.loss_fn(output_perturbed, dice_y)
        model.zero_grad()
        
        loss.backward()
        
        X_pert = X_pert + X_pert.grad.detach().sign() * self.epsilon
        X_pert = torch.clamp(X_pert.clone().detach().cuda(),min=0, max=1)
        X_pert.requires_grad = True
        
        return X_pert
    
    
def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)



class BIM(object):
    def __init__(self, model=None, epsilon=None, loss_fn=None, organs=None, device=None):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.device = device
        self.organs = organs
        
    def perturb(self, X, y, epsilons=None, niters=None, alpha=1):
        model = self.model
        DEVICE = self.device
        ORGANS = self.organs
        
        if epsilons is not None:
            self.epsilon = epsilons   
        use_cuda = torch.cuda.is_available()   
        X_pert = X.clone().detach().cuda()
        X_pert.requires_grad = True
        y = y.to(DEVICE, dtype=torch.long)
        
        for i in range(niters):
            output_perturbed = None
            output_perturbed = model(X_pert.cuda())
            output_perturbed = F.softmax(output_perturbed, dim=1)

            dice_y = F.one_hot(y, num_classes=len(ORGANS)).movedim(-1,1).float()
            dice_y = dice_y.squeeze().reshape(output_perturbed.size()) # 모양 안맞을 때 맞춰줌
            loss = self.loss_fn(output_perturbed, dice_y)
            model.zero_grad()

            loss.backward()

            X_pert = X_pert + X_pert.grad.detach().sign() * alpha
            X = X.cuda()
            X_pert = where(X_pert>X+self.epsilon, X+self.epsilon, X_pert)
            X_pert = where(X_pert<X-self.epsilon, X-self.epsilon, X_pert)
            
            X_pert = torch.clamp(X_pert.clone().detach().cuda(),min=0, max=1)
            X_pert.requires_grad = True
        
        return X_pert
    
    
class PGD(object):
    def __init__(self, model=None, epsilon=None, loss_fn=None, organs=None, device=None):
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.device = device
        self.organs = organs
        
    def perturb(self, X, y, epsilons=None, niters=7,attack_lr=25.0/255.0):
        model = self.model
        DEVICE = self.device
        ORGANS = self.organs
        
        if epsilons is not None:
            self.epsilon = epsilons   
        use_cuda = torch.cuda.is_available()   
        X_pert = X.clone().detach().cuda()
        X_pert.requires_grad = True
        y = y.to(DEVICE, dtype=torch.long)
        
        for step in range(niters):
            # model.zero_grad()
            output_perturbed = None
            output_perturbed = model(X_pert.cuda())
            output_perturbed = F.softmax(output_perturbed, dim=1)

            dice_y = F.one_hot(y, num_classes=len(ORGANS)).movedim(-1, 1).float()
            dice_y = dice_y.squeeze()
            loss = self.loss_fn(output_perturbed, dice_y)
            model.zero_grad()

            loss.backward()
            
            grad = X_pert.grad.detach().sign()
            X_pert = X_pert + attack_lr * grad
            X = X.cuda()
            X_pert = X+ torch.clamp(X_pert-X, min=-self.epsilon, max=self.epsilon)
            X_pert = X_pert.clone().detach().cuda()
            X_pert = torch.clamp(X_pert, min=0, max=1)
            X_pert.requires_grad = True
            
        return X_pert