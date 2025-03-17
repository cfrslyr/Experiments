from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
from . import *
from .encoder import Encoder
from .utils.mae_utils import *


class Finetune(nn.Module):
    def __init__(self, config: OrderedDict,
                 img_size=32,
                 img_channel=3,
                 patch_size=4,
                 num_classes=10,
                 ):
        super(Finetune, self).__init__()

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