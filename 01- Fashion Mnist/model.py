import torch

class my_model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding_mode='same')
    # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding_mode='same')
    self.fc1 = torch.nn.Linear(784, 256)
    self.fc2 = torch.nn.Linear(256, 128)
    self.fc3 = torch.nn.Linear(128, 64)
    self.fc4 = torch.nn.Linear(64, 10)

  def forward(self, x):
    x = x.reshape((x.shape[0], 784))

    x = self.fc1(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.2, train=True)

    x = self.fc2(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.2, train=True)

    x = self.fc3(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.2, train=True)

    x = self.fc4(x)
    x = torch.softmax(x, dim=1)
    return x