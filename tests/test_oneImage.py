
from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

import sys
sys.path.append('/home/rsa-key-20230225/PycharmProjects/vit-pytorch/vit-pytorch')
from vit_pytorch.vit import ViT

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 1
epochs = 2000
lr = 3e-4
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'

os.makedirs('data', exist_ok=True)
# %%
train_dir = 'Data_oneImage/train'
test_dir = 'Data_oneImage/test'

# %%
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
# %%
print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")



labels = [path.split('/')[-1].split('.')[0] for path in train_list]
# random_idx = np.random.randint(1, len(train_list), size=9)
# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
#
# for idx, ax in enumerate(axes.ravel()):
#     img = Image.open(train_list[idx])
#     ax.set_title(labels[idx])
#     ax.imshow(img)

# train_list, valid_list = train_test_split(train_list,
#                                           test_size=0.2,
#                                           stratify=labels,
#                                           random_state=seed)
#%%
print(f"Train Data: {len(train_list)}")
# print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")



#Image augmentation
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


#Load dataset
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label

#%%
train_data = CatsDogsDataset(train_list, transform=train_transforms)
img,label = train_data[0]
img = img.unsqueeze(0)
# valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)
#%%
#train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
# valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)
#%%
#print(len(train_data), len(train_loader))

# print(len(valid_data), len(valid_loader))


model = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 2,
    dim = 128,
    depth = 12,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)



#Training
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
#%%
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    #for data, label in tqdm(train_loader):
    #data = data.to(device)
    #label = label.to(device)

    output = model(img)
    label = torch.tensor(label).reshape((1,))
    loss = criterion(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = (output.argmax(dim=1) == label).float().mean()
    epoch_accuracy += acc
    epoch_loss += loss

    print(f"epoch accuracy:{epoch_accuracy}, los: {loss}, epoch loss:{epoch_loss}\n")

    # with torch.no_grad():
    #     epoch_val_accuracy = 0
    #     epoch_val_loss = 0
    #     for data, label in valid_loader:
    #         #data = data.to(device)
    #         #label = label.to(device)
    #
    #         val_output = model(data)
    #         val_loss = criterion(val_output, label)
    #
    #         acc = (val_output.argmax(dim=1) == label).float().mean()
    #         epoch_val_accuracy += acc / len(valid_loader)
    #         epoch_val_loss += val_loss / len(valid_loader)

    # print(
    #     f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    # )
