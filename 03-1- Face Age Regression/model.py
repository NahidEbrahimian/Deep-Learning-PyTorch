import torch
import torch.nn as nn
import torchvision.models as models

def model(device):
  # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = models.resnet50(pretrained=True)
  in_fetures = model.fc.in_features
  model.fc = nn.Linear(in_fetures, 1)
  model = model.to(device)
  
  # This freezes layers 1-6 in the total 10 layers of Resnet50
  ct = 0
  for child in model.children():
      ct += 1
      if ct < 7:
          for param in child.parameters():
              param.requires_grad = False

  return model