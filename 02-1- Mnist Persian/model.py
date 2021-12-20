import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv2d(3, 32, (3, 3), (1, 1), (1, 1)) 
    self.conv2 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
    self.conv3 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
    # self.conv4 = nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))

    self.fc1 = nn.Linear(128*8*8 ,128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    conv1 = F.relu(self.conv1(x))
    conv1 = F.max_pool2d(conv1, kernel_size=(2, 2))

    conv2 = F.relu(self.conv2(conv1))
    conv2 = F.max_pool2d(conv2, kernel_size=(2, 2))

    conv3 = F.relu(self.conv3(conv2))
    conv3 = F.max_pool2d(conv3, kernel_size=(2, 2))

    # conv4 = F.relu(self.conv4(conv3))
    # conv4 = F.max_pool2d(conv4, kernel_size=(2, 2))

    flatten = torch.flatten(conv3, start_dim=1)
    flatten = torch.dropout(flatten, 0.2, train=True)

    fc1 = self.fc1(flatten)
    fc1 = torch.dropout(fc1, 0.4, train=True)

    fc2 = self.fc2(fc1)
    output = torch.softmax(fc2, dim=1)

    return output