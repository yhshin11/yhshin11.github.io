---
# jupyter:
#   jupytext:
#     cell_metadata_filter: incorrectly_encoded_metadata,-all
#     cell_metadata_json: true
#     text_representation:
#       extension: .md
#       format_name: markdown
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
title: ""
# permalink: /notes/quick-start-guide/
# excerpt: "How to quickly install and setup Minimal Mistakes for use with GitHub Pages."
last_modified_at: 2020-07-27
# redirect_from:
#   - /theme-setup/
# layout: single
classes: wide
author_profile: false
share: false
toc: true
---

# PyTorch

## Quickstart
Link to [quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
### Imports
```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
```
### Download sample dataset
```python
from torchvision import datasets
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
### Iterate over data wrapped by `DataLoader`
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
```
### Create model
To define a neural network:
- Define a class that inherits from `nn.Module`.
- Define layers in `__init__()`
- Define forward propagation in `forward()`

```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```
### Training
Define a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions) and an [optimizer](https://pytorch.org/docs/stable/optim.html).
```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and backpropagates the prediction error to adjust the model’s parameters.
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad() # Set all gradients to zero
        loss.backward() # Backpropagate gradients
        optimizer.step() # Update parameters
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

Also check the model’s performance against the test dataset to ensure it is learning.
```python
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # Sets the model to evaluation mode, equivalent to `model.train(False)`
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad(): # Disable gradient storing/calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

Conduct training over desired number of epochs.
```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```
### Saving/loading models
```python
# Save model state dictionary (containing parameters)
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
# Load model
model = NeuralNetwork() # First create a blank model instance with the same structure
model.load_state_dict(torch.load("model.pth")) # Load parameters into blank model instance
```
The model can now be used to make predictions:
```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

### Get activation of intermediate layers
```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cl1 = nn.Linear(25, 60)
        self.cl2 = nn.Linear(60, 16)
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model = MyModel()
model.fc2.register_forward_hook(get_activation('fc2'))
x = torch.randn(1, 25)
output = model(x)
print(activation['fc2'])
```

## Cheatsheet
[Link](https://pytorch.org/tutorials/beginner/ptcheat.html) to official cheatsheet
  

## Misc
```python
################
# Tensor
################
torch.tensor(data)
torch.from_numpy(np_array)
torch.ones(shape)
torch.ones_like(t)
torch.zeros(shape)
torch.zeros_like(t)
torch.rand(shape)
torch.rand_like(t)
torch.randn(shape)
torch.arange()
torch.linspace()
torch.tanh()
torch.svd(t)
t.shape
t.dtype
t.device
t.T
t.to(device)
t.numpy()
t.cpu()

################
# Tensor operations
################
# Matrix multiplication
t @ t.T
torch.matmul()
# Element-wise multiplication
t * t
torch.mul()
torch.pow()
t.norm() # Matrix/vector norm - Deprecated! Use `torch.linalg.norm()` instead
# Get python numeric value from single value `Tensor`
t.item()
# Return new copy vs in-place operation
t.t()
t.t_()
t.add()
t.detach()
# Reshaping
torch.squeeze(t, dim=None) # Squeeze dimensions of size one
torch.unsqueeze(t, dim) # Insert dimension of size one
torch.cat() # Concatenates along *existing* dimension
torch.stack() # Creates and stacks along *new* dimension
################
# Autograd-related
################
# Disable autograd
with torch.no_grad(): t.add_(t2)
t = torch.tensor(np.arange(5), requires_grad=True)
t.requires_grad_(True)
t.backward()
t.grad_fn
t.grad
```

### Model API
```python
################
# Model API
################
Y = model(X)
model.weight.data
model.to(device)
```

### `torch.nn` API
```python
nn.Sequential()
################
# Layers
################
nn.Linear()
nn.Tanh()
################
# Loss functions
################
nn.CrossEntropyLoss()
nn.MSELoss()
```


### `torch.optim`
```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # built-in L2
```

### Misc
```python
################
# Linear algebra
################
linalg.vector_norm()
linalg.matrix_norm()
################
# Setting random seed
################
import torch
torch.manual_seed(seed)
import random
random.seed(0)
```

# Pytorch Lightning
## Introduction
See [here](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html).
The documentation page in general has tons of useful tricks.

## Rapid prototyping template
Adapted from [here](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html#starter-templates).
Use `bolts` for dummy dataset.
Remove if unnecessary.

### Install packages if necessary
```python
# ! pip install pytorch-lightning
# ! pip install pytorch-lightning-bolts
# ! pip install torchmetrics
```

### Basic template
```python
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.datasets import DummyDataset

################
# Define data
################
train = DummyDataset((1, 28, 28), (1,))
train = DataLoader(train, batch_size=32)

val = DummyDataset((1, 28, 28), (1,))
val = DataLoader(val, batch_size=32)

test = DummyDataset((1, 28, 28), (1,))
test = DataLoader(test, batch_size=32)

################
# Define model
################
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

################
# Train
################
model = LitAutoEncoder()
trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=20)
trainer.fit(model, train, val)

################
# Test
################
trainer.test(test_dataloaders=test)

```

### Visualize training results
Lightning automatically logs metrics when you can `self.log()`.
Default logger is TensorBoard which you can call on the command line with:
`tensorboard --logdir lightning_logs/`
or use the following ipython extension to view logs in notebook.
```python
%load_ext tensorboard
%tensorboard --logdir lightning_logs/
```
### Other optional features to consider
- Callbacks for better monitoring, saving checkpoints, etc.
- Config object
- LR Scheduler
- Accelerator, e.g. GPU
- Lightning `DataModule`


## Deep learning template
Generate from [here](https://github.com/PyTorchLightning/deep-learning-project-template).
This comes with `setup.py`, `setup.cfg` and `requirements.txt`.

## Misc. comments
- `LightningModule.training_step(self, batch, batch_idx)` must return `loss` as a tensor or a dictionary containing an entry `'loss'`.
  - This is so that backpropagation can be performed on the loss object.
- `training_step`, `validation_step`, etc can also return quantities to be calculated at epoch-level. `training_epoch_end` is passed a list of all batch-level outputs, and can be overridden for epoch-level operations. See [here](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-operations) for example.
- Regarding `Callbacks`: `torch.no_grad()` and `pl_module.eval()` is unnecessary in validation steps since Lightning turns it on automatically.

# Baseline config for new experiments
```python
class cfg:
    # Flags
    DEBUG = False
    # Data
    root_dir = 'data'
    # General settings
    seed = 42 # Seed for random number generators
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Trainer settings
    train_split = 0.8
    batch_size = 16
    num_workers = 4 # For DataLoader
    lr = 1e-3 # Starting learning rate
    # Model-specific
    model_name = 'tf_efficientnetv2_s_in21k' # Pretrained model
    pretrained = True
    img_size = 300 # Resize image to size
    # Store useful derived variables and objects here
    num_classes = None
    steps_per_epoch = None
    label_encoder = None # Cultivar label to integer
    data_size = None # Number of images for train + validation
```
