# ML TL;DR

## Models

### DALL-E and CLIP

### ViT (Vision Transformers)

### BERT

### GPT-3

### Swin Transformer

### EfficientNet

### NFNet

### Inception

### ResNet

### R-CNN, Fast R-CNN, Faster R-CNN
2-stage models (first propose region, then classify/locate) used for object detection.

R-CNN:
Uses external region proposal method. Runs CNN on each each region.

Fast R-CNN:
Runs shared CNN on proposed regions.

Faster R-CNN:
Uses a CNN-based Region Proposal Network (RPN), with a classifier and box regressor on top.

### Mask R-CNN
Model used for instance segmentation, i.e. object detection + semantic segmentatiion.
Uses an FPN (Feature Pyramid Network) on top of Faster R-CNN.

RoIAlign:
A modification of RoI Pool to reduce effect of quantization error when downsampling features.
Important since Mask R-CNN produces pixel-wise masks.

## Concepts

### Test-time augmentation (TTA)
The idea of using augmentation when making predictions with a trained model,
in order to allow the model to make predictions for multiple different versions of each image in the test dataset.
The predictions on the augmented images can be averaged, which can result in better predictive performance.
See [here](https://machinelearningmastery.com/how-to-use-test-time-augmentation-to-improve-model-performance-for-image-classification/)
for some more discussion and examples of papers that use TTA.

Related: Multi-crop evaluation.

### Learning rate schedulers
[Article](https://spell.ml/blog/lr-schedulers-and-adaptive-optimizers-YHmwMhAAACYADm6F) about learning rate schedulers and adaptive optimizers
TL;DR: Use Adam/AdamW if you want something insensitive to choice of learning rate.
Use OneCycleLR with tuned maximum learning rate for faster learning.
