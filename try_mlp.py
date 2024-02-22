# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:48:06 2024

@author: Bella
"""

import numpy as np
import netCDF4 as nc
import json
import xarray as xr
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf

BASE_PATH_DATA = '../data/skogsstyrelsen/'
img_paths_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_train.npy')))
img_paths_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_val.npy')))
img_paths_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_names_test.npy')))
json_content_train = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_train.npy'), allow_pickle=True))
json_content_val = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_val.npy'), allow_pickle=True))
train_label = list(np.load(os.path.join(BASE_PATH_DATA, "skogs_gts_train.npy")))
val_label = list(np.load(os.path.join(BASE_PATH_DATA, "skogs_gts_val.npy")))
#json_content_test = list(np.load(os.path.join(BASE_PATH_DATA, 'skogs_json_test.npy'), allow_pickle=True))

BAND_NAMES = ['b01', 'b02', 'b03', 'b04', 'b05', 'b06', 'b07', 'b08', 'b8a', 'b09', 'b11', 'b12']

def compose_img(img_index, img_path):
    img = xr.open_dataset(img_path[img_index])
    

    band_list = []
    for band_name in BAND_NAMES:
    	band_list.append(getattr(img, band_name).values/ 10000)  # 10k division
    img = np.concatenate(band_list, axis=0)
    img = np.transpose(img, [1,2,0])
    img = np.fliplr(img).copy()
    img = np.flipud(img).copy()
    
    if img.shape[1] == 21:
        # new_col = img[:,-1,:]
        # np.hstack((img, new_col[:,np.newaxis]))
        img = img[:,:20,:]
    if img.shape[0] == 21:
        img = img[:20,:,:]
    return img

network = nn.Sequential(
    nn.Linear(20*20*12, 100), 
    nn.ReLU(),
    nn.Linear(100, 1), 
    nn.Sigmoid()
)

# Initialize the optimizer
# In addition to changing optimizer you can try to change other parameters like learning rate (lr)
optimizer = optim.SGD(network.parameters(), lr=0.01)

# Initialize the loss function

loss_function = nn.BCELoss()
def train_one_epoch(data_path, labels):
    running_loss = 0.

    for i in range(len(data_path)):
        label = labels[i]
        label = torch.tensor([[label]], dtype=torch.float32)

        # Reshape the images 
        image = compose_img(i, data_path)
        image = image.reshape(1, 20*20*12)
        image = torch.tensor(image, dtype=torch.float32)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        pred = network(image)

        # Compute the loss and its gradients
        loss = loss_function(pred, label)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss/len(data_path)

EPOCHS = 20

best_vloss = 1
train_loss = []
vali_loss = []
for epoch in range(EPOCHS):
    # print('EPOCH {}:'.format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    network.train(True)

    train_loss_avg = train_one_epoch(img_paths_train, train_label)
    train_loss.append(train_loss_avg)

    # running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    # network.eval()
    # # Disable gradient computation and reduce memory consumption.
    # for i in range(len(img_paths_val)):
    #     # Every data instance is an input + label pair
    #     #inputs, labels = data
    #     vlabel = val_label[i]
    #     vlabel = torch.tensor([[vlabel]], dtype=torch.float32)
        

    #     # Reshape the images to a single vector (28*28 = 784)
    #     vimage = compose_img(i, img_paths_val)
    #     vimage = vimage.reshape(1, 20*20*12)
    #     vimage = torch.tensor(vimage, dtype=torch.float32)
    #     vpred = network(vimage)

    #     vloss = loss_function(vpred, vlabel)
    #     running_vloss += vloss
        
    # avg_vloss = running_vloss/len(img_paths_val)
    # vali_loss.append(avg_vloss)

    
#     print("Train loss:", train_loss_avg)
#     print("Validation loss", avg_vloss)
#     epoch += 1
# # Plot the training loss per epoch
# plt.plot(range(1,EPOCHS+1),train_loss[:], label = "train loss")
# plt.plot(range(1,EPOCHS+1),vali_loss[:], label = "validation loss")
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.show()

def accuracy(path = img_paths_val, label = val_label):
    correct = 0
    for i in range(len(path)):
        vlabel = label[i]
        vlabel = torch.from_numpy(np.array([vlabel])).type(torch.LongTensor)       

        # Reshape the images to a single vector (28*28 = 784)
        vimage = compose_img(i, img_paths_val)
        vimage = vimage.reshape(1, 20*20*12)
        vimage = torch.tensor(vimage, dtype=torch.float32)
        vpred = network(vimage)
        if vpred >0.5 and vlabel == 1: # sigmoid threshold set as 0.5
            correct += 1
        elif vpred <0.5 and vlabel == 0:
            correct += 1
    return correct/len(path)

val_accuracy = accuracy(path = img_paths_val, label = val_label)       
        