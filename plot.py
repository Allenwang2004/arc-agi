import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------------------------------------------------------
# ARC-style palette (feel free to replace with your own)
# 0-9 integers → 10 RGB triples
ARC_PALETTE = np.array([
    [  0,   0,   0],   # 0 black
    [255,   0,   0],   # 1 red
    [  0, 255,   0],   # 2 green
    [255, 255,   0],   # 3 yellow
    [  0,   0, 255],   # 4 blue
    [255,   0, 255],   # 5 magenta
    [  0, 255, 255],   # 6 cyan
    [255, 255, 255],   # 7 white
    [128, 128, 128],   # 8 gray
    [128,   0,   0],   # 9 dark-red (example)
], dtype=np.uint8)
ARC_CMAP = ListedColormap(ARC_PALETTE / 255.0, name='arc')

# -------------------------------------------------------------------
def plot_grid(grid, title='', ax=None, cmap=ARC_CMAP):
    """
    Display an integer-labelled colour grid on the given Matplotlib axis.

    Parameters
    ----------
    grid  : 2-D ndarray of ints 0-9
    title : str
    ax    : matplotlib.axes.Axes or None
    cmap  : matplotlib.colors.Colormap
    """
    if ax is None:
        ax = plt.gca()

    ax.imshow(grid, interpolation='nearest', cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])