# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import logging
import os
import sys
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
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

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# print_config()



# set this in your environment or previous cell to wherever IXI is downloaded and extracted
RunFolder = "/HippoDense/runFolder"
DatasetDir = "/Datasets/01_P_Classification_all_hippo"
Labels = []
Images = []
NotFitImages = []
for root, dirs, files in os.walk(os.path.abspath(DatasetDir)):
    for file in files:
        if file.split("_")[-3] == "L":
            Labels.append(2)
            Images.append(os.path.join(root, file))
        elif file.split("_")[-3] == "E":
            Labels.append(1)
            Images.append(os.path.join(root, file))
        elif file.split("_")[-2] == "Normal":
            Labels.append(0)
            Images.append(os.path.join(root, file))
        elif file.split("_")[-2] == "Dementia":
            Labels.append(3)
            Images.append(os.path.join(root, file))
        else:
            NotFitImages.append(file)
            # print(file)
print("{}: images cannot be used.".format(len(NotFitImages)))
# # labels = np.load(
#     '/workspace/monai/MONAI-tutorials/3d_classification/01_P_Classification_all_labels.npy')
NumClasses = len(np.unique(Labels))
# ClassLabels = np.array(Labels)
if len(Images)==len(Labels):
    print("{}: total images prepared with {} different classes".format(len(Images), NumClasses))
# images


# Represent labels in one-hot format for binary classifier training,
# BCEWithLogitsLoss requires target to have same shape as input
pin_memory = torch.cuda.is_available()

labels = torch.nn.functional.one_hot(torch.as_tensor(Labels)).float()
print(labels.shape)
# Define transforms
train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((224, 256, 208)), RandRotate90(), EnsureType()])
# train_transforms = Compose([ScaleIntensity(), AddChannel(), EnsureType()])

val_transforms = Compose(
    [ScaleIntensity(), AddChannel(), Resize((224, 256, 208)), EnsureType()])
# val_transforms = Compose(
    # [ScaleIntensity(), AddChannel(), EnsureType()])

# Define nifti dataset, data loader
check_ds = ImageDataset(image_files=Images, labels=labels, transform=train_transforms)
check_loader = DataLoader(check_ds, batch_size=1, num_workers=64, pin_memory=pin_memory)

im, label = monai.utils.misc.first(check_loader)
print(type(im), im.shape, label, label.shape)


# create a training data loader
train_ds = ImageDataset(image_files=Images[:-66], labels=labels[:-66], transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=64, pin_memory=pin_memory)

# create a validation data loader
val_ds = ImageDataset(image_files=Images[-66:], labels=labels[-66:], transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=32, pin_memory=pin_memory)

# Create DenseNet121, CrossEntropyLoss and Adam optimizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer(k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
"""

model = monai.networks.nets.DenseNet121(
    spatial_dims=3, in_channels=1, out_channels=NumClasses).to(device)

# loss_function = torch.nn.CrossEntropyLoss()
loss_function = torch.nn.BCEWithLogitsLoss()  # also works with this data

optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
max_epochs = 300

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar(" - train_loss", loss.item(), epoch_len * epoch + step)

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()

        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            with torch.no_grad():
                val_outputs = model(val_images)
                value = torch.eq(val_outputs.argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()

        metric = num_correct / metric_count
        metric_values.append(metric)

        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch + 1
            timeNow = time.strftime("%Y%m%d_%H%M%S")
            bestModelPath = os.path.join(RunFolder, "best_metric_model_classification3d_array_" + timeNow + ".pth")
            torch.save(model.state_dict(), bestModelPath)
            print("saved new best metric model")

        print(f"Current epoch: {epoch+1} current accuracy: {metric:.4f} ")
        print(f"Best accuracy: {best_metric:.4f} at epoch {best_metric_epoch}")
        writer.add_scalar("val_accuracy", metric, epoch + 1)

print(f"Training completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()

import pickle
with open('clf.pickle', 'wb') as f:
    pickle.dump(model, f)
# modelPath = "/workspace/monai/run_3d_classification/best_metric_model_classification3d_array20220415_063943.pt"
# model = torch.load(modelPath)