from opt_einsum import contract as einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


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
        attn_output_weights = einsum('bsnd, ndh, brnh->bsnr', x, self.attn_weight, x) / self.head_dim
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.attn_dropout(attn_output_weights)
        attn_output = einsum('bsnr, ndh, brnd->bsnh', attn_output_weights, self.value_weight, x)
        bsz, seq_len, _, _ = attn_output.shape
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        attn_output = self.out_project(attn_output)
        return attn_output
    

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
        x = self.pseudo_mhsa(x)
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
        attn_output_weights = self.softmax(attn_output_weights)
        attn_output_weights = self.attn_dropout(attn_output_weights)

        attn_output = torch.matmul(attn_output_weights, x)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        attn_output = self.out_project(attn_output)
        return attn_output
    

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

        self.cls_attn_probs = self.pseudo_mhsa.cls_attn_probs

    def forward(self, inputs):
        x = self.ln_1(inputs)
        x = self.pseudo_mhsa(x)
        x = self.drop(x)
        x = x + inputs
        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.drop(y)
        return y + x