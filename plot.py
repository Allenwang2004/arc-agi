import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# ARC 顏色映射
ARC_COLORS = [
    '#000000',  # 0: 黑色
    '#0074D9',  # 1: 藍色
    '#FF4136',  # 2: 紅色
    '#2ECC40',  # 3: 綠色
    '#FFDC00',  # 4: 黃色
    '#AAAAAA',  # 5: 灰色
    '#F012BE',  # 6: 粉紅色
    '#FF851B',  # 7: 橙色
    '#7FDBFF',  # 8: 青色
    '#870C25',  # 9: 紫色
]
arc_cmap = ListedColormap(ARC_COLORS)


def visualize_task(challenges,task_id):
    """視覺化一個 ARC 任務"""
    task = challenges[task_id]
    n_train = len(task['train'])
    
    fig, axes = plt.subplots(n_train, 2, figsize=(6, 3 * n_train))
    if n_train == 1:
        axes = axes.reshape(1, -1)
    
    for i, pair in enumerate(task['train']):
        axes[i, 0].imshow(pair['input'], cmap=arc_cmap, vmin=0, vmax=9)
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(pair['output'], cmap=arc_cmap, vmin=0, vmax=9)
        axes[i, 1].set_title(f'Output {i+1}')
        axes[i, 1].axis('off')
    
    plt.suptitle(f'Task: {task_id}')
    plt.tight_layout()
    plt.show()