from vsa import SimpleSSPSpace
import nengo_spa as spa
import numpy as np
import random
from torch import nn
import torch

# 設定隨機種子
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_DIMENSIONS = 4096
N_COLOURS = 10
MAX_GRID_SIZE = 30
LENGTH_SCALE = 0.25

DOMAIN_BOUNDS = np.array([[-MAX_GRID_SIZE / 2, MAX_GRID_SIZE / 2], 
                          [-MAX_GRID_SIZE / 2, MAX_GRID_SIZE / 2]])

SSP_SPACE = SimpleSSPSpace(
    ssp_dim=N_DIMENSIONS,
    domain_bounds=DOMAIN_BOUNDS,
    length_scale=LENGTH_SCALE,
    seed=SEED,
)

# ============================================
# 建立 SP 空間（用於離散概念）
# ============================================
SP_SPACE = spa.Vocabulary(
    N_DIMENSIONS,
    pointer_gen=np.random.RandomState(SEED),
)

# 定義顏色標籤
colour_tags = ["BLACK", "BLUE", "RED", "GREEN", "YELLOW", "GREY", "PINK", "ORANGE", "CYAN", "MAROON"]
feature_tags = ["COLOUR", "CENTRE", "SHAPE"]

# 填充 SP 空間
SP_SPACE.populate(";".join(colour_tags + feature_tags))

# 取得顏色的 SP 向量
COLOUR_SPS = np.array([SP_SPACE[tag].v for tag in colour_tags])