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
def plot_grid(grid, title='', ax=None, cmap=ARC_CMAP, show_grid=True, grid_color='black', grid_linewidth=1):
    """
    Display an integer-labelled colour grid with square cells on the given Matplotlib axis.

    Parameters
    ----------
    grid  : 2-D ndarray of ints 0-9
    title : str
    ax    : matplotlib.axes.Axes or None
    cmap  : matplotlib.colors.Colormap
    show_grid : bool, whether to show grid lines
    grid_color : str, color of grid lines
    grid_linewidth : float, width of grid lines
    """
    if ax is None:
        ax = plt.gca()

    # Convert to numpy array if not already
    grid = np.array(grid)
    
    # Display the grid
    im = ax.imshow(grid, interpolation='nearest', cmap=cmap, vmin=0, vmax=9)
    
    # Set aspect ratio to 'equal' to ensure square cells
    ax.set_aspect('equal')
    
    # Add grid lines to separate cells
    if show_grid:
        # Set up grid lines
        height, width = grid.shape
        
        # Major ticks at cell boundaries
        ax.set_xticks(np.arange(-0.5, width, 1), minor=False)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=False)
        
        # Grid lines
        ax.grid(True, which='major', color=grid_color, linewidth=grid_linewidth, alpha=0.8)
        
        # Remove tick labels but keep the grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        # Remove all ticks if no grid
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Set title
    ax.set_title(title, fontsize=12, pad=10)
    
    # Remove extra whitespace around the plot
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    
    return im


def plot_task_examples(examples, max_examples=3, figsize=(12, 8)):
    """
    Plot multiple input-output pairs from ARC task examples.
    
    Parameters
    ----------
    examples : list of dicts
        Each dict should have 'input' and 'output' keys with 2D arrays
    max_examples : int
        Maximum number of examples to display
    figsize : tuple
        Figure size for the plot
    """
    n_examples = min(len(examples), max_examples)
    
    fig, axes = plt.subplots(2, n_examples, figsize=figsize)
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_examples):
        example = examples[i]
        
        # Plot input
        plot_grid(example['input'], f'Input {i+1}', ax=axes[0, i])
        
        # Plot output
        plot_grid(example['output'], f'Output {i+1}', ax=axes[1, i])
    
    plt.tight_layout()
    return fig


def plot_prediction_comparison(input_grid, true_output, predicted_output, figsize=(12, 4)):
    """
    Compare true output vs predicted output for a single test case.
    
    Parameters
    ----------
    input_grid : 2D array
        Input grid
    true_output : 2D array
        Ground truth output
    predicted_output : 2D array
        Model's predicted output
    figsize : tuple
        Figure size for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    plot_grid(input_grid, 'Input', ax=axes[0])
    plot_grid(true_output, 'True Output', ax=axes[1])
    plot_grid(predicted_output, 'Predicted Output', ax=axes[2])
    
    # Add a visual indicator if prediction is correct
    if np.array_equal(true_output, predicted_output):
        axes[2].set_title('Predicted Output ✓', color='green', fontweight='bold')
    else:
        axes[2].set_title('Predicted Output ✗', color='red', fontweight='bold')
    
    plt.tight_layout()
    return fig