---
title: "Plotting"
last_modified_at: 2020-07-27
classes: wide
author_profile: false
share: false
toc: true
---

# Standard boiler plate
```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# For 3d plots
from mpl_toolkits import mplot3d
```

# Matplotlib

### Cheatsheet of methods
```python
# Plotting functions
ax.plot() # Simple 2D y-vs-x plots with lines or markers
ax.scatter() # Similar to ax.plot(), but with varying marker size/color
ax.imshow() # Display data as image, e.g. 2D heat map
# 3D plots
# ax.view_init() to set viewing angle
# Requires 3d axes object, e.g.
# ax = fig.add_subplot(projection='3d')
ax.plot3D()
ax.scatter3D()
ax.contour3D()
ax.plot_wireframe() # Wireframe plot
ax.plot_surface() # Surface plot
ax.plot_trisurf() # Triangulated surface plot for non-grid data
ax.fill_between() # Fill between two sets of y-values
# Annotations, labels, legends, etc
ax.legend()
ax.text() # Text annotations
ax.annotate() # Annotate with text+arrow
ax.add_patch() # Add a `Patch`. Use to draw low level geometric shapes
# Subplots
plt.subplot() # Single subplot
fig.add_subplot()
fig.subplots_adjust() # Adjust subplot spacing
plt.subplots() # Create grid of subplots
plt.GridSpec() # Use to create more complicated grids
plt.errorbar(x, y[, yerr, xerr, fmt, ecolor, ...])	# Plot y versus x as lines and/or markers with attached errorbars.

```

### Object-oriented vs Matlab-style API
Object-oriented API:
```python
fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x));
```

Matlab-style (state-ful) API:
```python
plt.plot(x, np.sin(x));
```
While most ``plt`` functions translate directly to ``ax`` methods (such as ``plt.plot()`` → ``ax.plot()``, ``plt.legend()`` → ``ax.legend()``, etc.), this is not the case for all commands.
In particular, functions to set limits, labels, and titles are slightly modified.
For transitioning between MATLAB-style functions and object-oriented methods, make the following changes:

- ``plt.xlabel()``  → ``ax.set_xlabel()``
- ``plt.ylabel()`` → ``ax.set_ylabel()``
- ``plt.xlim()``  → ``ax.set_xlim()``
- ``plt.ylim()`` → ``ax.set_ylim()``
- ``plt.title()`` → ``ax.set_title()``

In the object-oriented interface to plotting, rather than calling these functions individually, it is often more convenient to use the ``ax.set()`` method to set all these properties at once:

```python
ax = plt.axes()
ax.plot(x, np.sin(x))
ax.set(xlim=(0, 10), ylim=(-2, 2),
       xlabel='x', ylabel='sin(x)',
       title='A Simple Plot')
```
### Basic plots
#### Simple 2D plots with lines or markers
```python
# Axis limits
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)
plt.axis([-1, 11, -1.5, 1.5]) # [xmin, xmax, ymin, ymax]
plt.axis('tight')
plt.axis('equal')
plt.axis('auto')
# Title, labels, legends
plt.title("A Sine Curve")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.legend() # Combine with `label` argument for plot() method
# Explicit label specification for legends
ax.legend([line1, line2, line3], ['label1', 'label2', 'label3'])
# Line color
plt.plot(x, np.sin(x - 0), color='blue')        # specify color by name
plt.plot(x, np.sin(x - 1), color='g')           # short color code (rgbcmyk)
plt.plot(x, np.sin(x - 2), color='0.75')        # Grayscale between 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44')     # Hex code (RRGGBB from 00 to FF)
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) # RGB tuple, values 0 to 1
plt.plot(x, np.sin(x - 5), color='chartreuse'); # all HTML color names supported
# Line style
plt.plot(x, x + 0, linestyle='solid')
plt.plot(x, x + 1, linestyle='dashed')
plt.plot(x, x + 2, linestyle='dashdot')
plt.plot(x, x + 3, linestyle='dotted');
plt.plot(x, x + 4, linestyle='-')  # solid
plt.plot(x, x + 5, linestyle='--') # dashed
plt.plot(x, x + 6, linestyle='-.') # dashdot
plt.plot(x, x + 7, linestyle=':');  # dotted
# style+color
plt.plot(x, x + 0, '-g')  # solid green
plt.plot(x, x + 1, '--c') # dashed cyan
plt.plot(x, x + 2, '-.k') # dashdot black
plt.plot(x, x + 3, ':r');  # dotted red
# Markers
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
for marker in markers:
    plt.plot(rng.rand(5), rng.rand(5), marker,
             label="marker='{0}'".format(marker))
```

### Misc tips

#### Set style sheet

```python
plt.style.available # View available styles
plt.style.use('classic') # Set classic style
plt.style.use('seaborn-whitegrid')
plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
plt.style.use('bmh')
```

#### Save figure
```python
fig.savefig('my_figure.png')
```

#### Get current objects
```python
plt.gcf() # Get current figure
plt.gca() # Get current axes
```

#### Date tick labels
https://matplotlib.org/stable/gallery/text_labels_and_annotations/date.html

#### Specifying x, y location relative to data, axes, and figure
```python
ax.text(1, 5, ". Data: (1, 5)", transform=ax.transData)
ax.text(0.5, 0.1, ". Axes: (0.5, 0.1)", transform=ax.transAxes)
ax.text(0.2, 0.2, ". Figure: (0.2, 0.2)", transform=fig.transFigure);
```

#### Customizing ticks
See:  
https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html

#### Get histogram bins
```python
# Utility function to get n, bins of histogram from matplotlib axes object
def get_hist(ax):
    n,bins = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1-y0)
        bins.append(x0) # left edge of each bin
    bins.append(x1) # also get right edge of last bin
    return n,bins
```

#### List of named colors
https://matplotlib.org/stable/gallery/color/named_colors.html

#### Iterate over a grid of subplots
```python
fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
```

#### Display black and white image
```python
ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
```

#### Display text relative to axes coordinates
```python
ax.text(0.05, 0.05, str(digits.target[i]),
        transform=ax.transAxes, color='green')
```

#### Color-labeled scatter plot with color bar
```python
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);
```

#### Draw plain black arrow
```python
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
```

#### Annotate with image
```python
imagebox = offsetbox.AnnotationBbox(
    offsetbox.OffsetImage(image, cmap=cmap),
                            xy)
ax.add_artist(imagebox)
```

#### Annotate embedding with images from sample
```python
# TODO: Cleanup
def plot_components(data, model, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    proj = model.fit_transform(data)
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)

```

#### Draw ellipses to show 2d covariance
```python
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
```

#### Display image files in grid
```python
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[ pic_index-8:pic_index] 
               ]

next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
```

# Seaborn

## API Reference
https://seaborn.pydata.org/api.html

## Cheatsheet
``` python
# Figure-level methods
# (Each of the following can plot different kinds of plots via the `kind` keyword)
sns.relplot() # Relational plots
sns.displot() # Distribution plots
sns.catplot() # Categorical plots

# Matrix plots
sns.heatmap()
# Regression plots
sns.lmplot() # Produces output of both regplot and residplot
sns.regplot()
sns.residplot()

# Grids
# Multi-plot grids (generic)
sns.FacetGrid()
FacetGrid.map()
FacetGrid.map_dataframe()
# Pair-wise grids (between pairs of columns of a dataset)
pairplot()
PairGrid()
PairGrid.map()
PairGrid.map_diag()
PairGrid.map_offdiag()
PairGrid.map_lower()
PairGrid.map_upper()
# Joint grids (bivariate and univariate graphs between two variables)
jointplot()
JointGrid()
JointGrid.plot()
JointGrid.plot_joint()
JointGrid.plot_marginals()
```
