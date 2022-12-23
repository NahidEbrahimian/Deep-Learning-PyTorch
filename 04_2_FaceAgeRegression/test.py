import os
import torch
import torch.nn as nn
import torchvision
from model import *
from data_loader import *
from argparse import ArgumentParser
from google_drive_downloader import GoogleDriveDownloader as gdd


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
  test_data = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
  return train_data, test_data

def calc_loss(y_pred, labels):
    acc=torch.abs(y_pred - labels.data) / len(y_pred)
    return acc

model = model(device)
_, test_data = build_dataset(16, args.data_path)

wieght_path = './age_regression.pth'
if not os.path.exists(wieght_path):
    gdd.download_file_from_google_drive(file_id='1-WEH9FtTaOl7FmZrWZ9ETTDEK3h8QyC_',
                                        dest_path=wieght_path)
model.load_state_dict(torch.load(wieght_path,  map_location=device))
model.eval()

test_loss=0.0
for img, label in test_data:

    img = img.to(device).float()
    label = label.to(device).float()

    pred = model(img)
    test_loss += calc_loss(pred, label)

total_loss = test_loss / len(test_data)
print(f"test loss: {total_loss}")