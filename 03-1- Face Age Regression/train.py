import os
import torch
import torch.nn as nn
import torchvision
from model import *
from data_loader import *
from argparse import ArgumentParser
import wandb

parser = ArgumentParser()
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--data_path", type=str)
args = parser.parse_args()
device = args.device


def build_dataset(batch_size, data_path):
  train_data = data_loader(data_path)
  torch.manual_seed(0)
  train_dataset_size = int(0.8 * len(train_data))
  test_dataset_size = len(train_data) - train_dataset_size

  train_data, test_data = torch.utils.data.random_split(train_data, [train_dataset_size, test_dataset_size])
  train_data = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
  test_data = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

  return train_data, test_data

model = model(device)
train_data, _ = build_dataset(64, args.data_path)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.L1Loss()

for epoch in range(10):
  train_loss = 0.0
  for images, labels in train_data:

    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    images = images.float()
    preds = model(images)

    loss = loss_function(preds, labels.float())
    loss.backward()
    optimizer.step()
    train_loss += loss

  total_loss = train_loss / len(train_data)
  print(f"Epoch: {epoch+1}, Loss: {total_loss}")