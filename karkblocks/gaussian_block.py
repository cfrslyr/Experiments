from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from .pseudo_mhsa_block import Mlp


class Gaussian_Functional(autograd.Function):
    @staticmethod
    def forward(ctx, x, sigma):
        B, S, E = x.shape
        x_ = x.view(B, S, 1, E, 1)
        y_ = x.view(B, 1, S, 1, E)

        base = (x_ - y_).pow_(2)
        base = torch.exp( - base / (2 * sigma ** 2))
        
        ctx.save_for_backward(x, base)
        ctx.sigma = sigma
        return base.sum(dim=(-1, -2))

    @staticmethod
    def backward(ctx, grad_outputs):
        x, base = ctx.saved_tensors
        sigma = ctx.sigma
        B, S, E = x.shape
        x_ = x.view(B, S, 1, E, 1)
        y_ = x.view(B, 1, S, 1, E)

        term = (x_ - y_) / sigma ** 2
        (term.mul_(base)).mul_(grad_outputs.view(B, S, S, 1, 1))

        return term.sum(dim=(1,3)) - term.sum(dim=(2,4)), None


class Gaussian_Fitting(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 attn_droprate: float=0.05,
                 sigma: float=1.0
                 ):
        super(Gaussian_Fitting, self).__init__()

        self.sigma = sigma
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, f"model_dim should be divisible by num_heads."

        self.ln = nn.LayerNorm(self.head_dim)
        self.in_project = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(attn_droprate)
        self.out_project = nn.Linear(model_dim, model_dim)

    def forward(self, inputs):
        bsz, seq_len, _ = inputs.shape

        x = self.in_project(inputs)
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        x = self.ln(x)

        x_gaussian = x.view(bsz * self.num_heads, seq_len, self.head_dim)
        attn_weights = Gaussian_Functional.apply(x_gaussian, self.sigma) / self.head_dim
        attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, x)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_outputs = self.out_project(attn_outputs)

        return attn_outputs


class Gaussian_Block(nn.Module):
    def __init__(self, model_dim,
                 num_heads,
                 mlp_dim,
                 attn_droprate,
                 mlp_droprate,
                 droprate,
                 sigma: float=1.0
                 ):
        super(Gaussian_Block, self).__init__()
        self.mlp = Mlp(model_dim, mlp_dim, model_dim, mlp_droprate)
        self.gaussian_function = Gaussian_Fitting(model_dim, num_heads, attn_droprate, sigma)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(droprate)

    def forward(self, inputs):
        x = self.ln_1(inputs)
        x = self.gaussian_function(x)
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y