---
title: "Handling data"
last_modified_at: 2020-07-27
# redirect_from:
#   - /theme-setup/
toc: true
---

# Pytorch
## Generic imports
Remove/add as necessary
```python
# Environment setup
import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import sklearn.model_selection
import torch
import torchvision
import timm
from PIL import Image
from torchinfo import summary
import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy

from typing import Callable, Optional, Any, Tuple

```

## Preparing transforms
For images that don't require any preprocessing, we just need `transforms.ToTensor()` to convert from `PIL.Image` to `torch.tensor`.

The following template can be used if pre-processing or data augmentation is required.

```python
def make_transforms():
    # Standard transform for pre-trained torchvision model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomCrop(cfg.img_size),
        # transforms.RandomResizedCrop(cfg.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
    #     transforms.Resize(256),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return {
        'train': train_transform,
        'val': val_transform,
    }
```


## Reading train/val images from same directory and assigning labels from DataFrame
Normally we would just move the images into class directories, e.g.
```
root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png
```

Sometimes we don't want to move the files, or we don't have write access to data directory, e.g. on Kaggle kernels.

In cases like this, we create a custom `Dataset` and override the `__getitem__()` method.

See for example:
```python

class CultivarDataset(torchvision.datasets.VisionDataset):
    """Simple Dataset class to keep track of train/val images stored in the same directory"""
    def __init__(
        self,
        root: str, # Root directory that contains all files
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        sample_dict: pd.DataFrame = None, # DataFrame that holds all file names and labels
        # sample_dict['image'] holds file names
        # sample_dict['label'] holds encoded class labels, i.e. integer labels
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        # DataFrame to keep track of file names and labels (targets)
        self.sample_dict = sample_dict
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Retrieve image and target by looking up file name and label in sample_dict"""
        fname = self.sample_dict['image'].iloc[index]
        image_path = os.path.join(self.root, fname)
        target = self.sample_dict['label'].iloc[index]
        pil_image = Image.open(image_path).convert("RGB")
        
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target
    
    def __len__(self) -> int:
        return len(self.sample_dict)

```
