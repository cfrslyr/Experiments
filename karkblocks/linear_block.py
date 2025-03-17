from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.autograd as autograd
from .pseudo_mhsa_block import Mlp
import opt_einsum as oe
import math

class Linear_Fitting(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 attn_droprate: float = 0.05,
                 hidden_feature: int = 256
                 ):
        super(Linear_Fitting, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, f"model_dim should be divisible by num_heads."

        self.attn_weight = Parameter(torch.Tensor(hidden_feature, self.num_heads, self.head_dim, self.head_dim))
        nn.init.kaiming_uniform_(self.attn_weight)

        self.in_project = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(attn_droprate)
        self.ln = nn.Softmax(dim=-1)
        self.out_project = nn.Linear(model_dim, model_dim)


    def forward(self, x):
        bsz, seq_len, _ = x.shape

        x = self.in_project(x)
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        x = self.ln(x)
        
        attn_weights = oe.contract("bnse, oneh, bnrh-> bnso", x, self.attn_weight, x) / (self.head_dim * math.sqrt(seq_len))
        attn_weights = self.ln(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, x)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_outputs = self.out_project(attn_outputs)

        return attn_outputs
    


class Linear_Block(nn.Module):
    def __init__(self, model_dim,
                 num_heads,
                 mlp_dim,
                 attn_droprate,
                 mlp_droprate,
                 droprate,
                 hidden_feature
                 ):
        super(Linear_Block, self).__init__()
        self.mlp = Mlp(model_dim, mlp_dim, model_dim, mlp_droprate)
        self.pseudo_mhsa = Linear_Fitting(model_dim, num_heads, attn_droprate, hidden_feature)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(droprate)

    def forward(self, inputs):
        x = self.ln_1(inputs)
        x = self.pseudo_mhsa(x)
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y + x