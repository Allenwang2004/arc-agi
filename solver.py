from object_predictor import OperationPredictor, ParameterPredictor, compute_global_weights, compute_similarity_heuristic, cleanup_colour
from dataloader import prepare_training_data
from train import (
    train_operation_predictor_with_pair_weights,
    train_parameter_predictor_with_pair_weights
)
from object import N_DIMENSIONS, SSP_SPACE ,SP_SPACE, normalize, detect_objects, encode_object
import numpy as np

class ARCVSASolver:
    """
    完整的 ARC-VSA 求解器
    支援每對 input-output 配對各自的權重
    """
    
    def __init__(self):
        self.operation_predictors = {}
        self.parameter_predictors = {}
        self.global_weights = None
        self.pair_weights = []  # 每對配對的權重
    
    def train(self, task):
        """
        從訓練範例學習
        """
        # Step 1: 準備訓練資料
        input_objs, output_objs, ops = prepare_training_data(task)
        
        if not input_objs:
            return False

        print(f"Extracted {len(input_objs)} object-operation pairs")
        print(f"Operation distribution: {dict((op, ops.count(op)) for op in set(ops))}")

        # Step 2: 計算每對配對的權重
        weights_list, matched_pairs = compute_similarity_heuristic(input_objs, output_objs)
        self.pair_weights = weights_list
        
        # 計算全域權重（用於測試時）
        self.global_weights = compute_global_weights(input_objs, output_objs)

        print(f"\nweight of each pair:")
        for i, weights in enumerate(weights_list):
            print(f"  pair {i+1}: colour={weights[0]:.3f}, centre={weights[1]:.3f}, shape={weights[2]:.3f}")
        print(f"global weight: colour={self.global_weights[0]:.3f}, centre={self.global_weights[1]:.3f}, shape={self.global_weights[2]:.3f}")

        # Step 3: 為每種操作訓練 predictor（使用各自的權重）
        unique_ops = set(ops)
        
        for op in unique_ops:
            if op == 'identity':
                continue  # Identity 不需要 predictor
            
            # 訓練 Operation Predictor（使用全域權重）
            op_predictor = OperationPredictor(N_DIMENSIONS, self.global_weights)
            labels = [1.0 if o == op else 0.0 for o in ops]
            
            # 使用每對配對各自的權重進行訓練
            train_operation_predictor_with_pair_weights(
                op_predictor, input_objs, labels, weights_list, n_epochs=50
            )
            self.operation_predictors[op] = op_predictor
            
            # 訓練 Parameter Predictor
            if op == 'recolour':
                targets = [out_obj.colour_repr for in_obj, out_obj, o in zip(input_objs, output_objs, ops) if o == op]
                inputs = [in_obj for in_obj, o in zip(input_objs, ops) if o == op]
                pair_w = [w for w, o in zip(weights_list, ops) if o == op]
            elif op == 'recentre':
                targets = [out_obj.centre_repr for in_obj, out_obj, o in zip(input_objs, output_objs, ops) if o == op]
                inputs = [in_obj for in_obj, o in zip(input_objs, ops) if o == op]
                pair_w = [w for w, o in zip(weights_list, ops) if o == op]
            elif op == 'reshape':
                targets = [out_obj.shape_repr for in_obj, out_obj, o in zip(input_objs, output_objs, ops) if o == op]
                inputs = [in_obj for in_obj, o in zip(input_objs, ops) if o == op]
                pair_w = [w for w, o in zip(weights_list, ops) if o == op]
            else:
                continue
            
            if inputs:
                param_predictor = ParameterPredictor(N_DIMENSIONS, self.global_weights)
                train_parameter_predictor_with_pair_weights(
                    param_predictor, inputs, targets, pair_w, n_epochs=50
                )
                self.parameter_predictors[op] = param_predictor

        print(f"\nTraining complete! Operation Predictors: {list(self.operation_predictors.keys())}")
        return True
    
    def predict_operations(self, test_objects):
        """
        預測每個物件的操作（使用全域權重）
        """
        predictions = []
        
        for obj in test_objects:
            best_op = 'identity'
            best_prob = 0.5
            
            for op, predictor in self.operation_predictors.items():
                prob = predictor.predict(obj)
                if prob > best_prob:
                    best_prob = prob
                    best_op = op
            
            predictions.append((best_op, best_prob))
        
        return predictions
    

# ===========================================
# 精確解碼的關鍵：保留原始像素位置！
# ===========================================
# 
# VSA 向量（特別是 shape）無法精確解碼回像素位置，因為：
# 1. Bundle (加法) 是有損操作 - 多個 SSP 加在一起無法分離
# 2. SSP 編碼本身就是近似的
#
# 正確做法：物件同時保存 VSA 表示 + 原始像素資訊
# ===========================================

class ARCObjectWithPixels:
    """
    擴展的 ARCObject：同時保存 VSA 表示和原始像素資訊
    - VSA 表示：用於學習和預測「做什麼操作」
    - 原始像素：用於精確生成輸出
    """
    def __init__(self, colour_repr, centre_repr, shape_repr, 
                 original_mask, original_colour, grid_shape):
        # VSA 表示
        self.colour_repr = colour_repr
        self.centre_repr = centre_repr
        self.shape_repr = shape_repr
        
        # 原始像素資訊（這才是精確的！）
        self.original_mask = original_mask
        self.original_colour = original_colour
        self.grid_shape = grid_shape
        
        # 計算原始像素位置列表
        self.pixel_positions = []
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                if original_mask[i, j]:
                    self.pixel_positions.append((i, j))
        
        # 綁定版本（用於神經網路）
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
        return (
            np.dot(self.colour_repr, other.colour_repr),
            np.dot(self.centre_repr, other.centre_repr),
            np.dot(self.shape_repr, other.shape_repr)
        )
    
    def bundle_weighted(self, weights):
        return normalize(
            weights[0] * self.bound_colour +
            weights[1] * self.bound_centre +
            weights[2] * self.bound_shape
        )

def create_arc_objects_with_pixels(grid):
    """建立帶有原始像素資訊的 ARCObject"""
    grid = np.array(grid)
    detected = detect_objects(grid)
    objects = []
    
    for mask, colour in detected:
        colour_repr, centre_repr, shape_repr = encode_object(mask, colour, grid.shape)
        obj = ARCObjectWithPixels(
            colour_repr, centre_repr, shape_repr,
            original_mask=mask,
            original_colour=colour,
            grid_shape=grid.shape
        )
        objects.append(obj)
    
    return objects

def generate_output_grid_precise(input_grid, solver):
    """
    精確的輸出生成：
    1. 用 VSA 預測操作和參數
    2. 套用到「原始像素位置」上
    """
    input_grid = np.array(input_grid)
    output_grid = np.zeros_like(input_grid)
    
    # 取得輸入物件（帶原始像素）
    input_objects = create_arc_objects_with_pixels(input_grid)
    
    if not input_objects:
        return output_grid
    
    # 預測每個物件的操作
    predictions = solver.predict_operations(input_objects)
    
    for obj, (operation, prob) in zip(input_objects, predictions):
        # 決定新顏色
        if operation == 'recolour':
            param_predictor = solver.parameter_predictors.get('recolour')
            if param_predictor:
                pred_vec = param_predictor.predict(obj)
                new_colour, _ = cleanup_colour(pred_vec)
            else:
                new_colour = obj.original_colour
        else:
            # identity 或其他操作：保持原色
            new_colour = obj.original_colour
        
        # 用原始像素位置畫到輸出（這是精確的！）
        for (r, c) in obj.pixel_positions:
            output_grid[r, c] = new_colour
    
    return output_grid

def evaluate_solver_on_task(solver, task_id, challenges_data, solutions_data):
    """
    評估 solver 在單一任務上的表現（使用精確解碼）
    """
    task = challenges_data[task_id]
    task_solutions = solutions_data[task_id]
    
    results = []
    
    for i, test_example in enumerate(task['test']):
        test_input = np.array(test_example['input'])
        expected_output = np.array(task_solutions[i])
        
        try:
            # 使用精確方法
            predicted_output = generate_output_grid_precise(test_input, solver)
            
            if predicted_output.shape == expected_output.shape:
                accuracy = np.mean(predicted_output == expected_output)
                exact_match = np.array_equal(predicted_output, expected_output)
            else:
                accuracy = 0.0
                exact_match = False
                
            results.append({
                'task_id': task_id,
                'test_idx': i,
                'accuracy': accuracy,
                'exact_match': exact_match
            })
        except Exception as e:
            results.append({
                'task_id': task_id,
                'test_idx': i,
                'accuracy': 0.0,
                'exact_match': False,
                'error': str(e)
            })

    return results