import os
import numpy as np
import cv2
import torch
import torchvision
from config import *
from model import *
from argparse import ArgumentParser
from google_drive_downloader import GoogleDriveDownloader as gdd

parser = ArgumentParser()
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--input_img", type=str)
args = parser.parse_args()

device=torch.device(args.device)

transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0), (1)),
])

img = cv2.imread(args.input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (width, height))

tensor = transform(img).unsqueeze(0).to(device)

model = Model()
model = model.to(device)

wieght_path = './age_regression.pth'
if not os.path.exists(wieght_path):
    gdd.download_file_from_google_drive(file_id='1-aBnXZ-T0CQmVQY2p1esD0t1vdMDaxRw',
                                        dest_path=wieght_path)
model.load_state_dict(torch.load(wieght_path))
model.eval()

pred = model(tensor)

print(f"age predicted: {pred[0]}")