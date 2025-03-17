from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.parameter import Parameter
from torch import Tensor

import math
import opt_einsum as oe
from collections import OrderedDict

from karkblocks import *
from torchvision.models.vision_transformer import EncoderBlock

from .utils.mae_utils import *


class Mlp(nn.Module):
    def __init__(self, in_feature, mlp_feature, out_feature, mlp_droprate):
        super(Mlp, self).__init__()

        self.linear1 = nn.Linear(in_features=in_feature, out_features=mlp_feature)
        self.dropout = nn.Dropout(mlp_droprate)
        self.linear2 = nn.Linear(in_features=mlp_feature, out_features=out_feature)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.normal_(self.linear1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.normal_(self.linear2.bias, std=1e-6)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


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
        attn_weights_1 = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights_1)

        attn_outputs = torch.matmul(attn_weights, x)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_outputs = self.out_project(attn_outputs)

        return attn_outputs, attn_weights_1


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
        x, attn_probs = self.gaussian_function(x)
        self.attn_probs = attn_probs
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y


class Pseudo_MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 attn_droprate: float=0.05
                 ):
        super(Pseudo_MultiHeadSelfAttention, self).__init__()

        self.model_dim = model_dim
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, f"model_dim should be divisible by num_heads."
        self.proj_weight = Parameter(torch.Tensor(model_dim, num_heads, self.head_dim))
        self.attn_weight = Parameter(torch.Tensor(num_heads, self.head_dim, self.head_dim))
        self.value_weight = Parameter(torch.Tensor(num_heads, self.head_dim, self.head_dim))
        self.reset_parameters()
        self.attn_dropout = nn.Dropout(attn_droprate)
        self.softmax = nn.Softmax(dim=-1)
        self.out_project= nn.Linear(model_dim, model_dim)

    def reset_parameters(self):
        nn.init.normal_(self.proj_weight, std=1 / self.model_dim ** 0.5)
        nn.init.normal_(self.attn_weight, std=.02)
        nn.init.normal_(self.value_weight, std=.02)
    
    def forward(self, x):
        x = torch.einsum('bse, end->bsnd', x, self.proj_weight)
        attn_output_weights = oe.contract('bsnd, ndh, brnh->bsnr', x, self.attn_weight, x) / self.head_dim
        attn_output_weights_1 = self.softmax(attn_output_weights)
        attn_output_weights = self.attn_dropout(attn_output_weights_1)
        attn_output = oe.contract('bsnr, ndh, brnd->bsnh', attn_output_weights, self.value_weight, x)
        bsz, seq_len, _, _ = attn_output.shape
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        attn_output = self.out_project(attn_output)
        return attn_output, attn_output_weights_1.permute(0,2,1,3).contiguous()
    

class Pseudo_MHSA_Block(nn.Module):
    def __init__(self, model_dim,
                 num_heads,
                 mlp_dim,
                 attn_droprate,
                 mlp_droprate,
                 droprate
                 ):
        super(Pseudo_MHSA_Block, self).__init__()
        self.mlp = Mlp(model_dim, mlp_dim, model_dim, mlp_droprate)
        self.pseudo_mhsa = Pseudo_MultiHeadSelfAttention(model_dim, num_heads, attn_droprate)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(droprate)

    def forward(self, inputs):
        x = self.ln_1(inputs)
        x, attn_probs = self.pseudo_mhsa(x)
        self.attn_probs = attn_probs
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y + x


class Optimized_MHSA(nn.Module):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 attn_droprate: float = 0.05
                 ):
        super(Optimized_MHSA, self).__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, f"model_dim should be divisible by num_heads."

        self.attn_weight = Parameter(torch.Tensor(1, num_heads, self.head_dim, self.head_dim))
        nn.init.normal_(self.attn_weight, std=0.02)

        self.in_project = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(attn_droprate)
        self.softmax = nn.Softmax(dim=-1)
        self.out_project = nn.Linear(model_dim, model_dim)


    def forward(self, x: torch.Tensor):
        bsz, seq_len, _ = x.shape

        x = self.in_project(x)
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output_weights = torch.matmul(x, self.attn_weight)
        attn_output_weights = torch.matmul(attn_output_weights, x.transpose(-2, -1)) / self.head_dim
        attn_output_weights_1 = self.softmax(attn_output_weights)
        attn_output_weights = self.attn_dropout(attn_output_weights_1)


        attn_output = torch.matmul(attn_output_weights, x)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        attn_output = self.out_project(attn_output)
        return attn_output, attn_output_weights_1
    

class Optimized_MHSA_Block(nn.Module):
    def __init__(self, model_dim,
                 num_heads,
                 mlp_dim,
                 attn_droprate,
                 mlp_droprate,
                 droprate
                 ):
        super(Optimized_MHSA_Block, self).__init__()
        self.mlp = Mlp(model_dim, mlp_dim, model_dim, mlp_droprate)
        self.pseudo_mhsa = Optimized_MHSA(model_dim, num_heads, attn_droprate)
        self.ln_1 = nn.LayerNorm(model_dim)
        self.ln_2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(droprate)


    def forward(self, inputs):
        x = self.ln_1(inputs)
        x, attn_probs = self.pseudo_mhsa(x)
        self.attn_probs = attn_probs
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y + x


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
        attn_weights_1 = self.ln(attn_weights)
        attn_weights = self.attn_dropout(attn_weights_1)

        attn_outputs = torch.matmul(attn_weights, x)
        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_outputs = self.out_project(attn_outputs)

        return attn_outputs, attn_weights_1
    


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
        x, attn_probs = self.pseudo_mhsa(x)
        self.attn_probs = attn_probs
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y + x



class Encoder_Block(EncoderBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None  # 存储注意力权重

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        # 调用父类的 self_attention 层并保存权重
        x, attn_probs = self.self_attention(x, x, x, need_weights=True, average_attn_weights=False)
        self.attn_probs = attn_probs  # 保存到实例变量
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, config: OrderedDict, encoder=True):
        super(Encoder, self).__init__()
        
        prefix = "encoder" if encoder else "decoder"
        
        self.model_dim = config[f"{prefix}_model_dim"]
        self.hidden_dim = config[f"{prefix}_hidden_dim"]
        self.num_layers = config[f"{prefix}_layers"]
        block = config["model"]

        block_map = {
            'pseudo_mhsa': lambda: Pseudo_MHSA_Block(
                self.model_dim, config["num_heads"], config[f"{prefix}_mlp_dim"],
                config["attn_droprate"], config["mlp_droprate"], config.get("droprate", 0.05)
            ),
            'gaussian': lambda: Gaussian_Block(
                self.model_dim, config["num_heads"], config[f"{prefix}_mlp_dim"],
                config["attn_droprate"], config["mlp_droprate"], config.get("droprate", 0.05)
            ),
            'spline': lambda: Spline_Block(
                self.model_dim, config["num_heads"], config[f"{prefix}_mlp_dim"],
                config["attn_droprate"], config["mlp_droprate"], config.get("droprate", 0.05)
            ),
            'optimized_mhsa': lambda: Optimized_MHSA_Block(
                self.model_dim, config["num_heads"], config[f"{prefix}_mlp_dim"],
                config["attn_droprate"], config["mlp_droprate"], config.get("droprate", 0.05)
            ),
            'vit': lambda: Encoder_Block(
                config["num_heads"], self.model_dim, config[f"{prefix}_mlp_dim"],
                config.get("droprate", 0.05), config["attn_droprate"]
            ),
            'linear': lambda: Linear_Block(
                self.model_dim, config["num_heads"], config[f"{prefix}_mlp_dim"],
                config["attn_droprate"], config["mlp_droprate"], config.get("droprate", 0.05), hidden_feature=65
            ),
        }
        
        self.layers = nn.Sequential(OrderedDict([
            (f"layer_{i}", block_map[block]())
            for i in range(self.num_layers)
        ]))
        
        self.dropout = nn.Dropout(config.get("droprate", 0.05))
        self.norm = nn.LayerNorm(self.model_dim)

        self.attentions = []
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
            attn_probs = layer.attn_probs.detach().cpu().numpy()
            self.attentions.append(attn_probs)
        return self.norm(x)


class Finetune_v(nn.Module):
    def __init__(self, config: OrderedDict,
                 img_size=32,
                 img_channel=3,
                 patch_size=4,
                 num_classes=10,
                 ):
        super(Finetune_v, self).__init__()

        # Initialize
        self.patch_size = patch_size
        self.model_dim = config["encoder_model_dim"]

        self.grid_size = img_size // self.patch_size
        self.patch_nums = self.grid_size ** 2

        self.cls_token = Parameter(torch.zeros(1, 1, self.model_dim))

        # Embed
        self.pos_embed = Parameter(torch.Tensor(1, self.patch_nums + 1, self.model_dim))
        self.patch_embed = PatchEmbed(self.model_dim, img_size, img_channel, self.patch_size)
        
        # Encoder and Classifier
        self.encoder = Encoder(config)
        self.classifer = nn.Linear(self.model_dim, num_classes)

    
    def forward(self, images):
        x = self.patch_embed(images)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        cls_tokens = x[:, 0, :]

        return self.classifer(cls_tokens)