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
import pandas
pin_memory = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# print_config()

classes = ['Normal', 'Dementia']

image_folder_path = '/API_run/hippo_image/'
image_path = ['/API_run/hippo_image/01_P_0004_1_Dementia_hippo.nii']
result_csv_file = '/API_run/result/result.csv'
# test_Image_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((224, 224, 224)), RandRotate90(), EnsureType()])

test_Image_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((224, 224, 224)), EnsureType(), AddChannel()])
test_ds = ImageDataset(image_files=image_path, transform=test_Image_transforms, image_only=False)

print("image shape:", test_ds[0][0].shape)


import pickle

modelPickle = "/HippoDense/EfficientNet/EfficientNet_20220603_035507.pickle"
savedModel = open(modelPickle, 'rb')
trainedModel = pickle.load(savedModel)
savedModel.close()

gpu_image_array = test_ds[0][0].to(device)
with torch.no_grad():
    val_outputs = trainedModel(gpu_image_array)
    # value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
    # print(val_outputs, torch.argmax(val_outputs, dim=1))
    # label = torch.argmax(val_outputs, dim=1)

    _ , predicted_labels = torch.max(val_outputs, 1)
    # print(predicted_labels)
    # print('Predicted: ', ' '.join(f'{classes[predicted_labels]}'))
    pd = pandas.DataFrame(data=val_outputs, columns=classes)
    # print(pd)
    pd.to_csv(result_csv_file)











