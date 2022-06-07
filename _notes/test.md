---
title: "Testing"
last_modified_at: 2020-07-27
publish: False
---
Go to [pytorch](pytorch.md)

# Generic imports
Remove/add as necessary
<details>
<summary>TITLE</summary>


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

<details>