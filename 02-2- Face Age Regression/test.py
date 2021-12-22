import os
import torch
import torch.nn as nn
import torchvision
from model import *
from config import *
from data_loader import *
from argparse import ArgumentParser
from google_drive_downloader import GoogleDriveDownloader as gdd


parser = ArgumentParser()
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--data_path", type=str)
args = parser.parse_args()

train_data = data_loader(args.data_path)

torch.manual_seed(0)
train_dataset_size = int(0.8 * len(train_data))
test_dataset_size = len(train_data) - train_dataset_size
_, test_data = torch.utils.data.random_split(train_data, [train_dataset_size, test_dataset_size])
test_data = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)

device=torch.device(args.device)
model = Model()
model=model.to(device)

wieght_path = './age_regression.pth'
if not os.path.exists(wieght_path):
    gdd.download_file_from_google_drive(file_id='1-aBnXZ-T0CQmVQY2p1esD0t1vdMDaxRw',
                                        dest_path=wieght_path)
model.load_state_dict(torch.load(wieght_path))
model.eval()

def calc_loss(y_pred, labels):
    acc=torch.abs(y_pred - labels.data) / len(y_pred)
    return acc


test_loss=0.0
for img, label in test_data:

    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    test_loss += calc_loss(pred, label)

total_loss = test_loss / len(test_data)
print(f"test loss: {total_loss}")