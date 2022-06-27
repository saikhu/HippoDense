#! /opt/conda/bin/python

import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import DataLoader, ImageDataset
from monai.transforms import (
    AddChannel,
    Compose,
    RandRotate90,
    Resize,
    ScaleIntensity,
    EnsureType
)
import pickle
import time

pin_memory = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
print_config()



train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((224, 224, 224)), RandRotate90(), EnsureType()])

val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((224, 224, 224)), EnsureType()])
check_ds = ImageDataset(image_files=images, labels=labels, transform=train_transforms)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)






















