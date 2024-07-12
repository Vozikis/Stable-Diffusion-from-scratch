import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        