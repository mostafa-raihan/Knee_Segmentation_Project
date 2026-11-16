import os
import torch
import pandas as pd
from args import get_args
from dataset import KneeDataset
from torch.utils.data import DataLoader
from model import UNetLext
from trainer import train_model

# 1:We need arguments
args = get_args()

# 2:Loading the csv files
train_set = pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

# 3:Define our datasets
train_dataset = KneeDataset(train_set)
val_dataset = KneeDataset(val_set)

# 4:Data Loader
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
print("train images:", len(train_dataset), "| val images:", len(val_dataset))

# 5: Initialize the model
model = UNetLext(
    input_channels=1,
    output_channels=1,
    pretrained=False,
    path_weights='',
    path_pretrained='',
    restore_weights=False,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 6: Train
train_model(model, train_loader, val_loader, device)
