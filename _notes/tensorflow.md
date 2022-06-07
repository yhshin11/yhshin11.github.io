---
title: "TensorFlow"
excerpt: ""
last_modified_at: 2020-07-27
---
## Basic skeleton
```python
# Get data
xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
ys = xs * 0.5 + 0.5

# Define your model (input_shape for first layer must be defined manually)
model = tf.keras.models.Sequential(
  tf.keras.layers.Dense(units=1, input_shape=[1]),
  )

# Compile model with optimizer, loss function and metrics
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train model, optionally providing validation data
model.fit(xs, ys, epochs=1000)

# Predict with model
new_x = 7.0
prediction = model.predict([new_x])[0]
```
    
## Basic example with fashion MNIST
```python
# Load Fashion MNIST data
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Define single hidden layer network with softmax classifer
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile and fit
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)
```
## Callbacks
### Early stopping
```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): # Must be defined in metrics
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
model.fit(training_images, training_labels, epochs=5, callbacks=[myCallback()])
```

# Transfer learning
See [here](https://www.tensorflow.org/tutorials/images/transfer_learning) for details.

# Data 
## `tf.data.Dataset`
See [here](https://www.tensorflow.org/guide/data) for reference.

## Quick recipes for data preprocessing
See [here](https://www.tensorflow.org/guide/keras/preprocessing_layers#quick_recipes) for quick recipes for
- Image data augmentation
- Normalizing numerical features
- Encoding string categorical features via one-hot encoding
- Encoding integer categorical features via one-hot encoding
- Etc.

### `tf.data.Dataset` methods
```python
# Creates a Dataset of a step-separated range of values.
range(10) # Creates Dataset from 0 to 9
# Creates a Dataset with at most count elements from this dataset.
take( count, name=None)
# Creates a Dataset that prefetches elements from this dataset.
prefetch( buffer_size, name=None)
# Returns an iterator which converts all elements of the dataset to numpy.
# E.g. list( tf.data.Dataset.range(5).as_numpy_iterator() )  --> [0, 1, 2, 3, 4]
as_numpy_iterator()
# Combines consecutive elements of this dataset into batches.
batch( batch_size, drop_remainder=False, num_parallel_calls=None, deterministic=None, name=None)
# Maps map_func across this dataset and flattens the result.
flat_map( map_func, name=None )
# Returns a dataset of "windows". Each "window" is a dataset that contains a subset of elements of the input dataset.
window( size, shift=None, stride=1, drop_remainder=False, name=None )
# Combines consecutive elements of this dataset into padded batches.
padded_batch( batch_size, padded_shapes=None, padding_values=None, drop_remainder=False, name=None )
```
### Misc
```python
# Shuffling and getting padded batches
ds_series.shuffle(20).padded_batch(10)
```



## Image data
### Manual loading a single image
```python
from tensorflow.keras.preprocessing.image import img_to_array, load_img
img = load_img(img_path, target_size=(300, 300))  # PIL image
x = img_to_array(img)  # Numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 300, 300, 3)
x /= 255 # Scale by 1/255
```
### Loading from directory
```python
# Create datasets from directory
# Labels are inferred from subdirectory by default
# See doc page for details:
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
```
### Preprocessing
There are two ways to use preprocessing layers.
One is to apply preprocessing on the `Dataset` objects using `Dataset.map()`.
```python
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
```

The second method is to include the preprocessing in the model definition to simplify deployment, e.g.
```python
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255), # Preprocessing
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
```

### Example
```python
# Define preprocessing steps
IMG_SIZE = 180
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets.
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
              num_parallel_calls=AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(1000)
  # Batch all datasets.
  ds = ds.batch(batch_size)
  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)
  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)
```

### More fine grain control
Use tf.data.
See [official tutorial](https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control)

### Loading and preprocessing with ImageDataGenerator (deprecated)
```python
# FIXME: ImageDataGenerator is deprecated and not recommended for new code.
# See here for details:
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './horse-or-human/',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

# Misc
### Prefetch data for better throughput
See for details: https://www.tensorflow.org/guide/data_performance
```python
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
```

### One-hot encoding
```python
# Converts a class vector (integers) to binary class matrix.
tf.keras.utils.to_categorical(y, num_classes=None, )
```

### Reset global state
```python
# Use to reset global state, e.g. when creating many models in a loop.
tf.keras.backend.clear_session()
```

### Visualizing intermediate layer outputs, e.g. convolutions and pooling
This is the Functional API
```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

The functional API can be used to define a model that outputs the outputs from intermediate layers
```python
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
```

Putting this together, we can visualize the convolution and pooling layer outputs
```python
import matplotlib.pyplot as plt
from tensorflow.keras import models

f, axarr = plt.subplots(3,4)

FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
```

#### Example
See [here](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_3_compacted_images.ipynb)
for complete example.
```python
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Prepare a random input image from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

# Scale by 1/255
x /= 255.0

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  
  if len(feature_map.shape) == 4:
    
    #-------------------------------------------
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    #-------------------------------------------
    n_features = feature_map.shape[-1]  # number of features in the feature map
    size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
    
    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    
    #-------------------------------------------------
    # Postprocess the feature to be visually palatable
    #-------------------------------------------------
    for i in range(n_features):
      x  = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std ()
      x *=  64
      x += 128
      x  = np.clip(x, 0, 255).astype('uint8')
      display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

    #-----------------
    # Display the grid
    #-----------------
    scale = 20. / n_features
    plt.figure( figsize=(scale * n_features, scale) )
    plt.title ( layer_name )
    plt.grid  ( False )
    plt.imshow( display_grid, aspect='auto', cmap='viridis' )
```

### Plotting metrics
Assuming metrics were included in `model.compile()`:
```python
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
```

### Some common pre-trained models and architectures
Keras has some common CNN architectures, e.g ResNet, InceptionV3, EfficientNet, etc.
See [`keras.applications` doc page](https://www.tensorflow.org/api_docs/python/tf/keras/applications) for details.