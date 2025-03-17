from collections import OrderedDict
import torch
import torch.nn as nn
from karkblocks import *
from torchvision.models.vision_transformer import EncoderBlock


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
            'vit': lambda: EncoderBlock(
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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.layers(self.dropout(x)))