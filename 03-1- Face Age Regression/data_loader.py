import os
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset, Dataset
# from config import *

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform):

        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def data_loader(data_dir):

  images = [] # X
  ages = [] # Y

  for image_name in os.listdir(data_dir)[0: 9000]:
      parts = image_name.split('_')
      ages.append(int(parts[0]))
      
      image = cv2.imread(os.path.join(data_dir, image_name))
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      images.append(image)

  images = pd.Series(images, name='Images')
  ages = pd.Series(ages, name='Ages')

  df = pd.concat([images, ages], axis=1)


  under_4 = []
  for i in range(len(df)):
    if df['Ages'].iloc[i] <= 4:
      under_4.append(df.iloc[i])

  under_4 = pd.DataFrame(under_4)
  under_4 = under_4.sample(frac=0.3)

  up_4 = df[df['Ages'] > 4]
  df = pd.concat([under_4, up_4])

  df = df[df['Ages'] < 90]

  X = []
  Y =[]

  for i in range(len(df)):
    width = height = 224
    df['Images'].iloc[i] = cv2.resize(df['Images'].iloc[i], (width, height))

    X.append(df['Images'].iloc[i])
    Y.append(df['Ages'].iloc[i])

  X= np.array(X)
  Y = np.array(Y)

  X = X.reshape((-1, X.shape[3], X.shape[1], X.shape[2]))
  Y = Y.reshape(Y.shape[0], 1)

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  dataset = TensorDataset(X, Y)

  transform = transforms.Compose([
      torchvision.transforms.ToPILImage(),
      transforms.RandomRotation(10),
      # transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0), (1))
  ])


  train_dataset = CustomTensorDataset(tensors=(X, Y), transform=transform)

  return train_dataset
