import os
import torch
import torchvision
from config import *
from model import *
from argparse import ArgumentParser
from google_drive_downloader import GoogleDriveDownloader as gdd


def cal_acc(y_pred, labels):
    _,prediction = torch.max(y_pred,1)
    acc=torch.sum(prediction == labels.data, dtype = torch.float64) / len(y_pred)
    return acc

parser = ArgumentParser()
parser.add_argument("--device", type=str)
args = parser.parse_args()

transform = torchvision.transforms.Compose([
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize((0), (1)),
])

test_set = torchvision.datasets.FashionMNIST('./test_data', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

device=torch.device(args.device)
model = my_model()
model=model.to(device)

wieght_path = 'fashion_mnist.pth'
if not os.path.exists(wieght_path):
    gdd.download_file_from_google_drive(file_id='14KEOG7q8uZftBlYtGLDLu0tz3h3xCdbx',
                                        dest_path=wieght_path)
    
model.load_state_dict(torch.load(wieght_path))
model.eval()

test_acc=0.0
for img, label in test_loader:

    img = img.to(device)
    label = label.to(device)

    pred = model(img)
    test_acc += cal_acc(pred, label)

total_acc = test_acc / len(test_loader)
print(f"test accuracy: {total_acc}")