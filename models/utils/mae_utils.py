import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, model_dim, image_size=32, in_channels=3, patch_size=8):
        super(PatchEmbed, self).__init__()
        self.model_dim = model_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.linearprojection = nn.Conv2d(in_channels, model_dim, patch_size, patch_size)
    
    def forward(self, x):
        x = self.linearprojection(x)
        x = x.view(-1, self.model_dim, self.num_patches)
        return x.permute(0,2,1)
    
class ImgEmbed(nn.Module):
    def __init__(self, model_dim, image_size=32, in_channels=3, patch_size=8):
        super(ImgEmbed, self).__init__()
        self.model_dim = model_dim
        self.n_h = image_size // patch_size
        self.n_w = image_size // patch_size
        self.inverse_projection = nn.ConvTranspose2d(model_dim, in_channels, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = x.view(-1, self.model_dim, self.n_h, self.n_w)
        x = self.inverse_projection(x)
        return x


def img_to_patch(imgs: torch.Tensor, patch_size: int):
    B, C, H, W = imgs.shape
    p = patch_size
    n_h = H // p
    n_w = W // p
    x = imgs.reshape(B, C, n_h, p, n_w, p)
    x = x.permute(0,2,4,3,5,1)
    return x.reshape(B, n_h*n_w, -1)

def patch_to_img(patches: torch.Tensor, img_channel=3):
    B, S, E = patches.shape
    C = img_channel
    n_h = n_w = int(S**0.5)
    p = int((E // C) ** 0.5)
    H = W = n_h * p
    imgs = patches.reshape(B, n_h, n_w, p, p, 3)
    imgs = imgs.permute(0,5,1,3,2,4)
    imgs = imgs.reshape(B, C, H, W)
    return imgs


def random_masking(x, mask_ratio):
    B, N, D = x.shape
    num_masked = int(mask_ratio * N)

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    ids_keep = ids_shuffle[:, num_masked:]

    mask = torch.zeros(B, N, device=x.device)
    mask[:, :num_masked] = 1

    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_kept, mask, ids_restore

def reconstruct_embed(x_kept, mask_tokens, ids_restore):
    D = x_kept.shape[2]
    x = torch.cat([mask_tokens, x_kept], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
    return x