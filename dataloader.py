from object_predictor import determine_operation
from object import create_arc_objects

# 準備訓練資料
def prepare_training_data(task):
    """
    從任務中準備訓練資料
    
    回傳:
        all_input_objects: 所有輸入物件
        all_output_objects: 對應的輸出物件
        operations: 每個物件對應的操作
    """
    all_input_objects = []
    all_output_objects = []
    operations = []
    
    for pair in task['train']:
        in_objs = create_arc_objects(pair['input'])
        out_objs = create_arc_objects(pair['output'])
        
        for in_obj in in_objs:
            # 找到最匹配的輸出物件
            best_sim = -float('inf')
            best_out_obj = None
            
            for out_obj in out_objs:
                total_sim = sum(in_obj.get_similarity_to(out_obj))
                if total_sim > best_sim:
                    best_sim = total_sim
                    best_out_obj = out_obj
            
            if best_out_obj:
                all_input_objects.append(in_obj)
                all_output_objects.append(best_out_obj)
                operations.append(determine_operation(in_obj, best_out_obj))
    
    return all_input_objects, all_output_objects, operations