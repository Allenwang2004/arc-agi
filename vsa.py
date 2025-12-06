import numpy as np

# ============================================
# 自己實作 SSP 空間（不需要 sspspace 套件）
# ============================================
class SimpleSSPSpace:
    """
    簡化版的 Spatial Semantic Pointer (SSP) 空間
    使用隨機傅立葉特徵來編碼 2D 座標
    """
    def __init__(self, ssp_dim, domain_bounds, length_scale, seed=0):
        self.ssp_dim = ssp_dim
        self.domain_bounds = domain_bounds
        self.length_scale = length_scale
        
        # 生成隨機頻率向量
        rng = np.random.RandomState(seed)
        # 使用 ssp_dim // 2 個頻率（因為每個頻率產生 cos 和 sin 兩個維度）
        n_freqs = ssp_dim // 2
        self.freqs = rng.randn(n_freqs, 2) / length_scale
        
    def encode(self, coords):
        """
        將 2D 座標編碼為 SSP 向量
        coords: [x, y] 或 [[x1, y1], [x2, y2], ...]
        """
        coords = np.atleast_2d(coords)
        # 計算相位: freqs @ coords.T -> (n_freqs, n_points)
        phases = self.freqs @ coords.T  # (n_freqs, n_points)
        
        # 使用 cos 和 sin 編碼
        cos_features = np.cos(2 * np.pi * phases)
        sin_features = np.sin(2 * np.pi * phases)
        
        # 交錯排列 cos 和 sin
        ssp = np.zeros((self.ssp_dim, coords.shape[0]))
        ssp[0::2, :] = cos_features
        ssp[1::2, :] = sin_features
        
        # 正規化
        ssp = ssp / np.linalg.norm(ssp, axis=0, keepdims=True)
        
        return ssp.T  # (n_points, ssp_dim)
    
    def bind(self, a, b):
        """
        綁定兩個向量（使用循環卷積的近似：元素相乘）
        這是一個簡化版本，實際 VSA 使用循環卷積
        """
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        
        # 使用 FFT 實現循環卷積
        result = np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
        return result
    
    def invert(self, a):
        """
        計算向量的逆（用於解綁定）
        對於循環卷積，逆就是反轉向量
        """
        a = np.atleast_2d(a)
        # 反轉除了第一個元素之外的所有元素
        return np.hstack([a[:, :1], a[:, 1:][:, ::-1]])