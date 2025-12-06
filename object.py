import skimage.measure
import numpy as np
from vsa import SimpleSSPSpace
import nengo_spa as spa
from config import N_DIMENSIONS, N_COLOURS, MAX_GRID_SIZE, LENGTH_SCALE, SSP_SPACE, SP_SPACE, COLOUR_SPS

def normalize(vector):
    """正規化向量為單位長度"""
    if (vector == 0).all():
        return vector.flatten()
    return vector.flatten() / np.linalg.norm(vector)

def detect_objects(grid):
    """
    使用 8-連通性偵測網格中的物件
    回傳: [(object_mask, colour), ...]
    """
    grid = np.array(grid)
    obj_grid, n = skimage.measure.label(grid, background=0, return_num=True)
    
    objects = []
    for i in range(1, n + 1):
        mask = (obj_grid == i)
        indices = np.where(mask)
        colour = int(grid[indices[0][0], indices[1][0]])
        objects.append((mask, colour))
    
    return objects

def encode_object(object_mask, object_colour, grid_shape):
    """
    將一個物件編碼為三個 VSA 向量
    
    參數:
        object_mask: 布林遮罩，表示物件佔據的像素
        object_colour: 物件的顏色 (0-9)
        grid_shape: 網格的形狀 (n_rows, n_cols)
    
    回傳:
        colour_repr: 顏色表示 (N_DIMENSIONS,)
        centre_repr: 中心表示 (N_DIMENSIONS,)
        shape_repr: 形狀表示 (N_DIMENSIONS,)
    """
    n_rows, n_cols = grid_shape
    
    # Step 1: 顏色表示 - 直接查表
    colour_repr = COLOUR_SPS[object_colour].flatten()
    
    # Step 2: 計算中心和位置表示
    position_repr = np.zeros(N_DIMENSIONS)
    min_x, max_x = MAX_GRID_SIZE, -MAX_GRID_SIZE
    min_y, max_y = MAX_GRID_SIZE, -MAX_GRID_SIZE
    
    for i in range(n_rows):
        for j in range(n_cols):
            if object_mask[i, j]:
                # 轉換為以網格中心為原點的座標系
                x = j - (n_cols - 1) / 2
                y = (n_rows - 1) / 2 - i
                
                # 更新邊界
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
                
                # Bundle: 累加每個像素的 SSP 編碼
                position_repr += SSP_SPACE.encode([x, y]).flatten()
    
    # 計算中心座標
    centre_x = (max_x + min_x) / 2
    centre_y = (max_y + min_y) / 2
    centre_repr = SSP_SPACE.encode([centre_x, centre_y]).flatten()
    centre_repr = normalize(centre_repr)
    
    # Step 3: 形狀表示 - bind 中心的逆，實現平移不變性
    shape_repr = normalize(
        SSP_SPACE.bind(position_repr, SSP_SPACE.invert(centre_repr)).flatten()
    )
    
    return colour_repr, centre_repr, shape_repr

class ARCObject:
    """表示一個 ARC 物件的 VSA 編碼"""
    
    def __init__(self, colour_repr, centre_repr, shape_repr):
        self.colour_repr = colour_repr
        self.centre_repr = centre_repr
        self.shape_repr = shape_repr
        
        # 建立綁定版本（用於學習）
        self.bound_colour = SSP_SPACE.bind(
            SP_SPACE["COLOUR"].v, normalize(colour_repr)
        ).flatten()
        self.bound_centre = SSP_SPACE.bind(
            SP_SPACE["CENTRE"].v, normalize(centre_repr)
        ).flatten()
        self.bound_shape = SSP_SPACE.bind(
            SP_SPACE["SHAPE"].v, normalize(shape_repr)
        ).flatten()
    
    def get_similarity_to(self, other):
        """計算與另一個物件的相似度（顏色, 中心, 形狀）"""
        return (
            np.dot(self.colour_repr, other.colour_repr),
            np.dot(self.centre_repr, other.centre_repr),
            np.dot(self.shape_repr, other.shape_repr)
        )
    
    def bundle_weighted(self, weights):
        """生成加權的物件表示"""
        return normalize(
            weights[0] * self.bound_colour +
            weights[1] * self.bound_centre +
            weights[2] * self.bound_shape
        )

def create_arc_objects(grid):
    """從網格建立 ARCObject 列表"""
    grid = np.array(grid)
    detected = detect_objects(grid)
    objects = []
    
    for mask, colour in detected:
        colour_repr, centre_repr, shape_repr = encode_object(mask, colour, grid.shape)
        objects.append(ARCObject(colour_repr, centre_repr, shape_repr))
    
    return objects