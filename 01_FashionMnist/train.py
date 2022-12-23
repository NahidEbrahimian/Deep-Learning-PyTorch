import torch
import torchvision
from model import *
from config import *
from argparse import ArgumentParser

def calc_acc(preds, labels):
  _, pred = torch.max(preds, 1)
  acc = torch.sum(pred == labels.data, dtype=torch.float64) / len(preds)
  return acc


parser = ArgumentParser()
parser.add_argument("--device", type=str)
args = parser.parse_args()

device = torch.device(args.device)
model = my_model()
model = model.to(device)
model.train(True)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0), (1)),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    # torchvision.transforms.Scale()
])

dataset = torchvision.datasets.FashionMNIST("./dataset", train=True, download=True, transform=transform)
train_data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
  train_loss = 0.0
  train_acc = 0.0
  for images, labels in train_data:
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    preds = model(images)

    loss = loss_function(preds, labels)
    loss.backward()

    optimizer.step()

    train_loss += loss
    train_acc += calc_acc(preds, labels)

  total_loss = train_loss / len(train_data)
  total_acc = train_acc / len(train_data)
  print(f"Epoch: {epoch+1}, Loss: {total_loss}, Accuracy: {total_acc}")
