import torch
import torch.nn as nn

def train_operation_predictor_with_pair_weights(predictor, input_objects, labels, weights_list, n_epochs=100, lr=0.01):
    """
    訓練 Operation Predictor（使用每對配對各自的權重）
    
    參數:
        predictor: OperationPredictor 實例
        input_objects: 輸入物件列表
        labels: 每個物件的標籤（0 或 1）
        weights_list: 每個物件對應的權重
        n_epochs: 訓練輪數
        lr: 學習率
    """
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for obj, label, weights in zip(input_objects, labels, weights_list):
            optimizer.zero_grad()
            
            # 使用該配對專屬的權重計算加權表示
            x = obj.bundle_weighted(weights)
            x = torch.tensor(x, dtype=torch.float32)
            
            # 前向傳播
            pred = predictor(x)
            target = torch.tensor([label], dtype=torch.float32)
            
            # 計算損失
            loss = criterion(pred, target)
            epoch_loss += loss.item()
            
            # 反向傳播
            loss.backward()
            optimizer.step()
        
        losses.append(epoch_loss / len(input_objects))
    
    return losses

def train_operation_predictor(predictor, input_objects, labels, n_epochs=100, lr=0.01):
    """
    訓練 Operation Predictor
    
    參數:
        predictor: OperationPredictor 實例
        input_objects: 輸入物件列表
        labels: 每個物件的標籤（0 或 1）
        n_epochs: 訓練輪數
        lr: 學習率
    """
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for obj, label in zip(input_objects, labels):
            optimizer.zero_grad()
            
            # 計算加權表示
            x = obj.bundle_weighted(predictor.weights.numpy())
            x = torch.tensor(x, dtype=torch.float32)
            
            # 前向傳播
            pred = predictor(x)
            target = torch.tensor([label], dtype=torch.float32)
            
            # 計算損失
            loss = criterion(pred, target)
            epoch_loss += loss.item()
            
            # 反向傳播
            loss.backward()
            optimizer.step()
        
        losses.append(epoch_loss / len(input_objects))
    
    return losses

def train_parameter_predictor(predictor, input_objects, target_params, n_epochs=100, lr=0.001):
    """
    訓練 Parameter Predictor（使用固定權重）
    使用 Cosine Similarity Loss
    """
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for obj, target in zip(input_objects, target_params):
            optimizer.zero_grad()
            
            # 計算加權表示
            x = obj.bundle_weighted(predictor.weights.numpy())
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            
            # 前向傳播
            pred = predictor.nn(x)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            
            # 使用 Cosine Similarity Loss（希望預測與目標相似）
            y = torch.ones(1)
            loss = criterion(pred, target_tensor, y)
            epoch_loss += loss.item()
            
            # 反向傳播
            loss.backward()
            optimizer.step()
        
        losses.append(epoch_loss / len(input_objects))
    
    return losses


def train_parameter_predictor_with_pair_weights(predictor, input_objects, target_params, weights_list, n_epochs=100, lr=0.001):
    """
    訓練 Parameter Predictor（使用每對配對各自的權重）
    使用 Cosine Similarity Loss
    """
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        
        for obj, target, weights in zip(input_objects, target_params, weights_list):
            optimizer.zero_grad()
            
            # 使用該配對專屬的權重計算加權表示
            x = obj.bundle_weighted(weights)
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            
            # 前向傳播
            pred = predictor.nn(x)
            target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            
            # 使用 Cosine Similarity Loss（希望預測與目標相似）
            y = torch.ones(1)
            loss = criterion(pred, target_tensor, y)
            epoch_loss += loss.item()
            
            # 反向傳播
            loss.backward()
            optimizer.step()
        
        losses.append(epoch_loss / len(input_objects))
    
    return losses