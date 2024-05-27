# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/03_research.ipynb.

# %% auto 0
__all__ = ['get_classes_from_frame', 'DogsSubsetDataset', 'collate_fn', 'get_fasterrcnn_model', 'train_faster_rcnn',
           'plot_evaluate_fasterrcnn_predictions', 'evaluate_fasterrcnn_classification_accuracy']

# %% ../nbs/03_research.ipynb 10
import cv2
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
from miniai.datasets import show_images
import shutil

# %% ../nbs/03_research.ipynb 13
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

# %% ../nbs/03_research.ipynb 14
def get_classes_from_frame(df, column=None):
    """gets the ['background'] + the rest of classes from a dataframe `df` with classes in column specified by `column`"""
    if column is None: column = 'category'
    classes = ['background'] + df['category'].unique().tolist()
    return classes

# %% ../nbs/03_research.ipynb 15
import os
import torch

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import functional as F
from sklearn.preprocessing import LabelEncoder

class DogsSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df
        self.classes = get_classes_from_frame(df)
        self.id2label = {i:c for i, c in enumerate(self.classes)}
        self.label2id = {c:i for i, c in enumerate(self.classes)}

    def _label2id(self, labels):
        return [self.label2id[l] for l in labels]

    def _id2labels(self, ids):
        return [self.id2label[i] for i in ids]
        
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img = F.convert_image_dtype(read_image(item['image']), torch.float)
        boxes = torch.as_tensor(item['bboxes'], dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        num_objs = boxes.shape[0]
        labels  = torch.ones((num_objs,), dtype=torch.int64) * self._label2id([item['category']])[0]
        image_id = idx
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        return img, target

    def __len__(self):
        return self.df.shape[0]

def collate_fn(data):
    images = [item[0] for item in data]
    images = torch.stack(images, dim=0)
    targets = [item[1] for item in data]
    return images, targets

# %% ../nbs/03_research.ipynb 16
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# %% ../nbs/03_research.ipynb 17
from tqdm.auto import tqdm
from . import engine
from .engine import train_one_epoch, evaluate

def train_faster_rcnn(model, df):
        # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    #device = 'cpu'
    print('Using device ', device)
    # our dataset has two classes only - background and person
    num_classes = len(get_classes_from_frame(df))
    # use our dataset and defined transformations
    dataset = DogsSubsetDataset(df)
    dataset_test = DogsSubsetDataset(df)

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=16, shuffle=False,
        collate_fn=collate_fn)

    # get the model using our helper function
    #model = get_fasterrcnn_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.01)
    # and a learning rate scheduler

    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=20,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 100

    for epoch in tqdm(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    torch.save(model.state_dict(), 'model-fasterrcnn.pt')
    return model

# %% ../nbs/03_research.ipynb 27
def plot_evaluate_fasterrcnn_predictions(model, dataset):
    """
    Plots the predictions of the faster-rcnn model and compares
    against ground truth labels in `dataset`. 
    """
    indices = torch.randperm(len(dataset)).tolist()[:16]
    subset = torch.utils.data.Subset(dataset, indices)
    # Get predictions
    dl = torch.utils.data.DataLoader(subset, batch_size=16, collate_fn=collate_fn)
    batch = next(iter(dl))
    model = model.eval()
    with torch.no_grad():
        preds = model(batch[0])
    images = [np.array(F.to_pil_image(t)) for t in batch[0]]
    boxes = [np.array(pred['boxes'][0] if len(pred['boxes']) else [0, 0, 0, 0]) for pred in preds] # Pick top box for each prediction
    predicted_labels = ds._id2labels([(pred['labels'][0].item() if len(pred['labels']) else 0) for pred in preds]) # Pick top label for each prediction
    actual_labels = ds._id2labels([b['labels'][0].item() for b in batch[1]])
    titles = [f'Actual: {actual}, predicted: {predicted}' for actual, predicted in zip(actual_labels, predicted_labels)]
    annotated_images = [annotated_image(im, [box]) for im, box in zip(images, boxes)]
    show_images(annotated_images, titles=titles, figsize=(30, 20))

# %% ../nbs/03_research.ipynb 29
def evaluate_fasterrcnn_classification_accuracy(model, ds):
    device = torch.device('cuda') if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    dl = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate_fn)
    predicted_labels = []
    actual_labels = []
    for batch in dl:
        model = model.eval()
        model = model.to(device)
        with torch.no_grad():
            preds = model(batch[0].to(device))
        predicted_labels += [(pred['labels'][0].cpu().item() if len(pred['labels']) else 0) for pred in preds]
        actual_labels += [b['labels'][0].cpu().item() for b in batch[1]]
    accuracy = np.mean(np.array(actual_labels) == np.array(predicted_labels))
    return accuracy
