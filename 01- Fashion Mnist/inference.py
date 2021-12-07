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
parser.add_argument("--device", type=str)
parser.add_argument("--input_img", type=str)
args = parser.parse_args()

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
device=torch.device(args.device)

transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0), (1)),
])

img = cv2.imread(args.input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))

tensor = transform(img).unsqueeze(0).to(device)
# tensor = tensor.to(device)

model = my_model()
model = model.to(device)

wieght_path = 'fashion_mnist.pth'
if not os.path.exists(wieght_path):
    gdd.download_file_from_google_drive(file_id='14KEOG7q8uZftBlYtGLDLu0tz3h3xCdbx',
                                        dest_path=model_path)
    
model.load_state_dict(torch.load(wieght_path))
model.eval()

pred = model(tensor)
pred = pred.cpu().detach().numpy()
pred = np.argmax(pred)
output = classes[pred]

print(f"model prediction: {output}", pred)