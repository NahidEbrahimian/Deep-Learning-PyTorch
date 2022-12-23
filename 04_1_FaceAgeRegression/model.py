import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1)) 
    self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
    self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
    self.conv4 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
    self.conv5 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))

    self.fc1 = nn.Linear(512*7*7 ,256)
    self.fc2 = nn.Linear(256 ,128)
    self.fc3 = nn.Linear(128 ,64)
    self.fc4 = nn.Linear(64, 1)

  def forward(self, x):
    conv1 = F.relu(self.conv1(x))
    conv1 = F.max_pool2d(conv1, kernel_size=(2, 2))

    conv2 = F.relu(self.conv2(conv1))
    conv2 = F.max_pool2d(conv2, kernel_size=(2, 2))

    conv3 = F.relu(self.conv3(conv2))
    conv3 = F.max_pool2d(conv3, kernel_size=(2, 2))

    conv4 = F.relu(self.conv4(conv3))
    conv4 = F.max_pool2d(conv4, kernel_size=(2, 2))

    conv5 = F.relu(self.conv5(conv4))
    conv5 = F.max_pool2d(conv5, kernel_size=(2, 2))

    flatten = torch.flatten(conv5, start_dim=1)
    flatten = torch.dropout(flatten, 0.2, train=True)

    fc1 = self.fc1(flatten)
    fc1 = torch.dropout(fc1, 0.2, train=True)

    fc2 = self.fc2(fc1)
    fc2 = torch.dropout(fc2, 0.3, train=True)

    fc3 = self.fc3(fc2)
    fc3 = torch.dropout(fc3, 0.4, train=True)

    output = self.fc4(fc3)
    return output