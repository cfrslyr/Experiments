import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.autograd as autograd
from .pseudo_mhsa_block import Mlp


# class Spline_Functional(autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         B, S, E = x.shape

#         x_ = x.view(B, S, 1, E, 1)
#         y_ = x.view(B, 1, S, 1, E)

#         Min = torch.min(x_, y_)
#         term = Min * torch.max(x_, y_)

#         ctx.save_for_backward(x)

#         term1 = 0.5 * (term * Min).sum(dim=(-1,-2))
#         term2 = 1/6 * Min.pow(3).sum(dim=(-1,-2))

#         return E ** 2 + term.sum(dim=(-1, -2)) + term1 - term2
    
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         x, = ctx.saved_tensors

#         B, S, E = x.shape

#         x_ = x.view(B, S, 1, E, 1)
#         y_ = x.view(B, 1, S, 1, E)
#         Min = torch.min(x_, y_)


#         grad_x = y_ * (1 + Min) - 0.5 * Min.pow(2)
#         grad_x = grad_x * grad_outputs.view(B, S, S, 1, 1)
#         grad_y = x_ * (1 + Min) - 0.5 * Min.pow(2)
#         grad_y = grad_y * grad_outputs.view(B, S, S, 1, 1)

#         grad_x = grad_x.sum(dim=(2,4)) + grad_y.sum(dim=(1,3))

#         return grad_x


class Spline_Functional(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B, S, E = x.shape
        x_ = x.view(B, S, 1, E, 1)
        y_ = x.view(B, 1, S, 1, E)

        Min = torch.min(x_, y_)
        term = torch.max(x_, y_).mul_(Min)

        term1 = (torch.clone(Min).mul_(term)).sum(dim=(-1, -2))
        torch.cuda.empty_cache()
        term1.mul_(0.5)

        term = torch.sum(term, dim=(-1, -2))

        term2 = Min.pow_(3).sum(dim=(-1,-2))
        torch.cuda.empty_cache()
        term2.mul_(1/6)

        result = ((term.add_(term1)).sub_(term2)).add_(E ** 2)

        ctx.save_for_backward(x)

        torch.cuda.empty_cache()

        return result
    
    @staticmethod
    def backward(ctx, grad_outputs):
        torch.cuda.empty_cache()

        x, = ctx.saved_tensors

        B, S, E = x.shape

        x_ = x.view(B, S, 1, E, 1)
        y_ = x.view(B, 1, S, 1, E)
        Min = torch.min(x_, y_)

        grad_x = (1 + Min).mul_(y_)
        grad_x = grad_x.sub_((Min.pow(2)).mul_(0.5))
        torch.cuda.empty_cache()
        grad_x = (grad_x.mul_(grad_outputs.view(B, S, S, 1, 1))).sum(dim=(2,4))
        torch.cuda.empty_cache()

        grad_y = (1 + Min).mul_(x_)
        grad_y = grad_y.sub_((Min.pow(2)).mul_(0.5))
        torch.cuda.empty_cache()
        grad_y = (grad_y.mul_(grad_outputs.view(B, S, S, 1, 1))).sum(dim=(1,3))
        torch.cuda.empty_cache()

        grad_x = grad_x.add_(grad_y)
        torch.cuda.empty_cache()

        return grad_x


class Spline_Fitting(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 attn_droprate: float=0.05
                 ):
        super(Spline_Fitting, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, f"model_dim should be divisible by num_heads."

        self.ln = nn.Softmax(-1)
        self.in_project = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(attn_droprate)
        self.out_project = nn.Linear(model_dim, model_dim)

    def forward(self, inputs):
        bsz, seq_len, _ = inputs.shape

        x = self.in_project(inputs)
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        x = x.view(-1, seq_len, self.head_dim)
        x = self.ln(x)

        attn_weights = Spline_Functional.apply(x) / self.head_dim
        attn_weights = self.ln(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        attn_outputs = torch.matmul(attn_weights, x)
        attn_outputs = attn_outputs.view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2).contiguous()
        attn_outputs = attn_outputs.view(bsz, seq_len, -1)
        
        attn_outputs = self.out_project(attn_outputs)

        return attn_outputs


class Spline_Block(nn.Module):
    def __init__(self, model_dim,
                 num_heads,
                 mlp_dim,
                 attn_droprate,
                 mlp_droprate,
                 droprate
                 ):
        super(Spline_Block, self).__init__()
        self.mlp = Mlp(model_dim, mlp_dim, model_dim, mlp_droprate)
        self.spline_function = Spline_Fitting(model_dim, num_heads, attn_droprate)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(droprate)

    def forward(self, inputs):
        x = self.ln_1(inputs)
        x = self.spline_function(x)
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y