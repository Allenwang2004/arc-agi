import numpy as np
import torch
import torch.nn as nn
from object import ARCObject, COLOUR_SPS, N_DIMENSIONS, SSP_SPACE, normalize

def compute_softmax(x):
    """計算 softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def compute_weight_for_pair(input_obj, output_obj):
    """
    計算單一 input-output 配對的權重
    
    回傳: weights = [w_colour, w_centre, w_shape]
    """
    sims = input_obj.get_similarity_to(output_obj)
    
    # 計算變化量（1 - 相似度）
    changes = np.array([
        1 - sims[0],  # 顏色變化
        1 - sims[1],  # 中心變化
        1 - sims[2],  # 形狀變化
    ])
    
    # 使用 softmax 轉換為權重（變化大的屬性更重要）
    weights = compute_softmax(changes * 5)  # 放大差異
    
    # 如果其中一個權重大於 0.7，則只使用該屬性
    if np.any(weights > 0.7):
        dominant_index = np.argmax(weights)
        weights = np.zeros(3)
        weights[dominant_index] = 1.0
    
    return weights

def compute_similarity_heuristic(input_objects, output_objects):
    """
    計算每對 input-output 物件的權重
    
    回傳: 
        weights_list: List of weights，每個元素對應一對 (input, output)
        matched_pairs: List of (input_obj, output_obj, weights) tuples
    """
    if not input_objects or not output_objects:
        return [], []
    
    matched_pairs = []
    weights_list = []
    
    for in_obj in input_objects:
        # 找到最匹配的輸出物件
        best_sim = -float('inf')
        best_out_obj = None
        
        for out_obj in output_objects:
            total_sim = sum(in_obj.get_similarity_to(out_obj))
            if total_sim > best_sim:
                best_sim = total_sim
                best_out_obj = out_obj
        
        if best_out_obj:
            # 計算這對配對的權重
            weights = compute_weight_for_pair(in_obj, best_out_obj)
            weights_list.append(weights)
            matched_pairs.append((in_obj, best_out_obj, weights))
    
    return weights_list, matched_pairs

def compute_global_weights(input_objects, output_objects):
    """
    計算全域權重（所有配對的平均，用於兼容舊程式碼）
    """
    weights_list, _ = compute_similarity_heuristic(input_objects, output_objects)
    
    if not weights_list:
        return np.array([1/3, 1/3, 1/3])
    
    # 計算平均權重
    avg_weights = np.mean(weights_list, axis=0)
    
    # 正規化
    avg_weights = avg_weights / avg_weights.sum()
    
    return avg_weights

# 定義操作類型
OPERATIONS = {
    'identity': 0,   # 不變
    'recolour': 1,   # 改變顏色
    'recentre': 2,   # 改變位置
    'reshape': 3,    # 改變形狀
}

def determine_operation(input_obj, output_obj, threshold=0.9):
    """
    根據相似度決定套用什麼操作
    """
    sims = input_obj.get_similarity_to(output_obj)
    
    # 所有都相似 -> Identity
    if all(s > threshold for s in sims):
        return 'identity'
    
    # 找出變化最大的屬性
    min_sim_idx = np.argmin(sims)
    
    if min_sim_idx == 0:
        return 'recolour'
    elif min_sim_idx == 1:
        return 'recentre'
    else:
        return 'reshape'
    
class OperationPredictor(nn.Module):
    """
    Operation Predictor: 預測物件應該套用哪個操作
    """
    
    def __init__(self, n_dimensions, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # 簡單的兩層神經網路
        self.nn = nn.Sequential(
            nn.Linear(n_dimensions, n_dimensions // 4),
            nn.GELU(),
            nn.Linear(n_dimensions // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.nn(x)
    
    def predict(self, arc_object):
        """預測這個物件是否要套用此操作"""
        x = arc_object.bundle_weighted(self.weights.numpy())
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(x).item()
        
class ParameterPredictor(nn.Module):
    """
    Parameter Predictor: 預測操作的具體參數
    輸出是一個 VSA 向量，需要用 cleanup 解碼
    """
    
    def __init__(self, n_dimensions, weights):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        # 輸出是 VSA 向量
        self.nn = nn.Sequential(
            nn.Linear(n_dimensions, n_dimensions),
            nn.GELU(),
            nn.Linear(n_dimensions, n_dimensions),
        )
    
    def forward(self, x):
        return self.nn(x)
    
    def predict(self, arc_object):
        """預測參數向量"""
        x = arc_object.bundle_weighted(self.weights.numpy())
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return self.forward(x).numpy()
        
def cleanup_colour(colour_ssp):
    """
    Cleanup: 將預測的向量解碼為具體顏色
    找到最相似的顏色向量
    """
    similarities = colour_ssp.flatten() @ COLOUR_SPS.T
    colour_index = np.argmax(similarities)
    colour_ssp = COLOUR_SPS[colour_index].flatten()
    return colour_index, colour_ssp