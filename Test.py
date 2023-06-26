# import libraries
from typing import Mapping, Any

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torchvision.transforms import transforms


class ClassificationModel(nn.Module):
  def __init__(self):
    super(ClassificationModel, self).__init__()
    self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = resnet152(weights=ResNet152_Weights)
    self.newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
    for p in self.newmodel.parameters():
         p.requires_grad_(False)
    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(64),
      nn.LazyBatchNorm1d(),
      nn.ReLU(),
      nn.LazyLinear(8)
    )

  def forward(self, image_batch):
    image_batch = self.normalize(image_batch)
    prediction = self.newmodel(image_batch)
    prediction = self.layers(prediction)
    return prediction

  def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        for p in self.newmodel.parameters():
            p.requires_grad_(False)
        self.layers.load_state_dict(state_dict)

if __name__ == '__main__':
  net = ClassificationModel()
  x = torch.rand(size = (3, 3, 224, 224))
  out = net(x)
  print(out)


