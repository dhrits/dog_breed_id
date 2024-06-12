# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_training.ipynb.

# %% auto 0
__all__ = ['DEFAULT_RANDOM_SEED', 'seed_basic', 'seed_torch', 'seed_everything', 'get_subsets', 'RegularizerCB',
           'get_classification_accuracy', 'get_classification_accuracy_ensembled']

# %% ../nbs/05_training.ipynb 4
from .data_preprocessing import read_csv_with_array_columns
from .research import get_classes_from_frame
from .benchmark import *
from miniai.learner import *
from miniai.init import *
from miniai.activations import *
from miniai.sgd import *
from miniai.datasets import show_images

import cv2
import fastcore.all as fc
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt

import shutil
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torcheval.metrics import MulticlassAccuracy
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import timm

# %% ../nbs/05_training.ipynb 5
import os 
import random
import numpy as np 

DEFAULT_RANDOM_SEED = 2021

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
import torch
def seed_torch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    seed_torch(seed)

# %% ../nbs/05_training.ipynb 10
def get_subsets(df, valid_size=0.1, random_state=DEFAULT_RANDOM_SEED):
    train_subset, valid_subset = train_test_split(df, test_size=valid_size, stratify=df['category'], random_state=random_state)
    valid_subset, test_subset = train_test_split(valid_subset, train_size=0.5, test_size=0.5, random_state=random_state)
    return train_subset, valid_subset, test_subset

# %% ../nbs/05_training.ipynb 15
class RegularizerCB(Callback):
    def __init__(self, alpha=0.01): fc.store_attr()
    
    def after_loss(self, learn):
        param_sum = sum([(p**2).sum() for p in learn.model.parameters()])
        param_sum *= self.alpha
        learn.loss += param_sum

# %% ../nbs/05_training.ipynb 29
def get_classification_accuracy(model, dl):
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    matches = []
    model.eval()
    model.to(device)
    for (imgs, labels) in dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
        preds = logits.argmax(-1)
        matches.extend((preds == labels).cpu().tolist())
    return np.mean(matches)


def get_classification_accuracy_ensembled(models, dl):
    models = models if isinstance(models, list) else [models]
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    matches = []
    [model.eval() for model in models]
    models = [model.to(device) for model in models]
    for (imgs, labels) in dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # Do bagging
            logits = sum([model(imgs) for model in models])
            logits = logits/float(len(models))
        preds = logits.argmax(-1)
        matches.extend((preds == labels).cpu().tolist())
    return np.mean(matches)
