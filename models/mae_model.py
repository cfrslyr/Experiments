from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from . import *
from .encoder import Encoder
from .utils.mae_utils import *

class MAE(nn.Module):
    def __init__(self, config: OrderedDict,
                 img_size=32,
                 img_channel=3,
                 patch_size=4,
                 mask_ratio=0.6,
                 ):
        super(MAE, self).__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        # Initialize
        model_dim = config["encoder_model_dim"]
        decoder_dim = config["decoder_model_dim"]
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # Initialize cls token and mask token
        self.cls_token = Parameter(torch.zeros(1, 1, model_dim))
        self.mask_token = Parameter(torch.zeros(1, 1, decoder_dim))

        # Encoder and decoder modules
        self.encoder = Encoder(config, encoder=True)
        self.decoder = Encoder(config, encoder=False)


        self.embed = PatchEmbed(model_dim, img_size, img_channel, patch_size)
        
        self.seq_num = self.embed.num_patches + 1
        self.seq_num_keep = self.seq_num - int((self.seq_num-1) * mask_ratio)
        
        self.pos_embed = Parameter(torch.Tensor(1, self.seq_num, model_dim))


        self.pos_embed_decoder = Parameter(torch.Tensor(1, self.seq_num, decoder_dim))
        self.decoder_embed = nn.Linear(model_dim, decoder_dim)

        self.img_recon = ImgEmbed(decoder_dim, img_size, img_channel, patch_size)

        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.pos_embed_decoder, std=0.02)

    def forward_encoder(self, images):
        inputs = self.embed(images)
        inputs = inputs + self.pos_embed[:,1:,:]
        cls_tokens = self.cls_token + self.pos_embed[:,:1,:]
        cls_tokens = cls_tokens.expand(inputs.shape[0], -1, -1)
        x, mask, ids_restore = random_masking(inputs, self.mask_ratio)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.encoder(x)

        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        cls_tokens, x_ = x[:, :1, :], x[:, 1:, :]
        mask_tokens = self.mask_token.expand(x_.shape[0], self.seq_num-self.seq_num_keep, -1)
        x_ = reconstruct_embed(x_, mask_tokens, ids_restore)
        inputs = torch.cat([cls_tokens, x_], dim=1)
        inputs = inputs + self.pos_embed_decoder
        inputs = self.decoder(inputs)
        inputs = self.img_recon(inputs[:, 1:, :])

        return inputs
    
    def forward_loss(self, imgs, imgs_restore, mask):
        loss = (imgs - imgs_restore) ** 2
        loss = img_to_patch(loss, self.patch_size)
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
        
    def forward(self, imgs):
        latents, mask, ids_restore = self.forward_encoder(imgs)
        imgs_restore = self.forward_decoder(latents, ids_restore)
        loss = self.forward_loss(imgs, imgs_restore, mask)

        return imgs_restore, loss