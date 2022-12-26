import torch
import torch.nn as nn
import torch.nn.functional as F

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=8, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=48, stride=stride) # 1*80 array
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(2 * n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(4 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(8 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(8 * n_channel, n_output)
        # self.dropout1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(4 * n_channel, n_output)
        # self.dropout2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.flatten(x, start_dim=1)
        # x = self.dropout1(x)
        x = self.fc1(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def accuracy(self, preds, labels):
        maxs, indices = torch.max(preds, 1)
        acc = torch.sum(indices == labels) / len(preds)
        return acc.cpu()