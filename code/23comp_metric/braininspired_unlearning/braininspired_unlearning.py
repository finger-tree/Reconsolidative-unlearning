import copy
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torch import nn
from torch import optim

# Standard absolute imports from the parent package
from .. import metric
from .. import surf
from .. import train_lib

"""
1. image similarity used to map cause
2. classification similarity used to map cause
3. for each layer in the res-net18, similarity flagged different cause in each layer
    -> therefore each layer would have a different cause, and each pass would have a different footprint
    -> -> a way to classify each layer of the resnet to a fixed number of causes, the causes
"""

"""
backwards is forwards
resnet classifies things, causes are learnt, from each layer, it learns to classify based on causes as well -> causes are like resolutions to dataset participation -> on unlearn -> it resolves to that dataset participation and removes the cause -> achieving unlearning.


"""
def braininspired_unlearning(
    retain_loader,
    forget_loader,
    val_loader,
    class_weights,
    original_model,
    print_accuracy=False,
):  
  """brain inspired unlearning.
  
  Input data format (from usecase in main.py):
  - Loaders (retain_loader, forget_loader, val_loader) provide samples as dictionaries.
  - Each sample contains: {'image': torch.Tensor, 'age_group': torch.Tensor}
  - 'image': input tensor of shape (batch_size, channels, height, width)
  - 'age_group': target labels tensor of shape (batch_size,)
  """
  del class_weights

  unlearned_model = copy.deepcopy(original_model)

  epochs = 1
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(
      unlearned_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs
  )

  unlearned_model.train()
  for _ in range(epochs):
    for sample in retain_loader:
      inputs = sample['image']
      targets = sample['age_group']
      inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

      optimizer.zero_grad()
      outputs = unlearned_model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
    scheduler.step()

    if print_accuracy:
      unlearned_model.eval()
      surf.compute_accuracy_surf(
          ['retain', 'forget', 'val'],
          [retain_loader, forget_loader, val_loader],
          unlearned_model,
          'Finetune model',
          print_=True,
      )
      unlearned_model.train()

  unlearned_model.eval()
  return unlearned_model



