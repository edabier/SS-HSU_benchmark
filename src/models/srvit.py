import torch
import torch.nn as nn
import math
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

"-------------------------------------------------------------------------------------"
"SRViT: Self-Supervised Relation-Aware Vision Transformer for Hyperspectral Unmixing"
"The work uses Python 3.8 and Pytorch 2.0 + cudnn 11.8"
"This file referred to some contents of tnt network and ContraNorm."
" (tnt: Transformer in Transformer)."
"(ContraNorm: A Contrastive Learning Perspective on Oversmoothing and Beyond)."
"-------------------------------------------------------------------------------------"
" Thanks for tnt's authors"


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def get_position_code_code(N):
    
    window = 3  # Int_indow size
    length = 7  # out_window size
    n_r = math.ceil(N / window)
    n_c = math.ceil(N / window)
    position_matrix = np.zeros((N, N))
    i = 1
    for x in range(n_r):
        for y in range(n_c):
            if x < (n_r - 1):
                if y < (n_c - 1):
                    position_matrix[x * window:(x * window + window), y * window:(y * window + window)] = i
                    i = i + 1
                else:
                    position_matrix[x * window:(x * window + window), y * window:] = i
                    i = i + 1
            else:
                if y < (n_c - 1):
                    position_matrix[x * window:, y * window:(y * window + window)] = i
                    i = i + 1
                else:
                    position_matrix[x * window:, y * window:] = i
                    i = i + 1
    num_position = n_r * n_c
    half = (length - 1) // 2
    position_fea = np.zeros((N + length, N + length))

    position_fea[half:half + N, half:half + N] = position_matrix
    position_dim = n_r * n_c
    position_code = np.zeros((N, N, n_r * n_c))
    position_vector = np.zeros((n_r * n_c))
    for i in range(N):
        for j in range(N):
            x = i + half
            y = j + half
            window_num = position_fea[(x-half):(x + half), (y-half):(y+half)]
            window_num = window_num.reshape(1, -1)
            window_num = window_num[window_num != 0] - 1
            unique_num, unique_count = np.unique(window_num, return_counts=True)
            unique_num = np.array(unique_num, dtype=int)
            position_vector[unique_num] = unique_count
            position_code[i, j, :] = position_vector
    return position_dim, position_code

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.n_h = n_h
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x_embed, position_embed):
        x_embed = x_embed.view(-1, self.n_h)
        position_embed = position_embed.view(-1, self.n_h)
        scores = self.f_k(x_embed, position_embed)
        return scores

def smooth_matrix(height, width):
    getGaussKernel=  width * height
    h = np.eye(getGaussKernel)
    for i in range(getGaussKernel):
        if i - width >= 0 and i + width <= getGaussKernel - 1 and i % width != 0 and (i + 1) % width != 0:
            h[i + 1, i] = -0.25
            h[i - 1, i] = -0.25
            h[i + width, i] = -0.25
            h[i - width, i] = -0.25
        elif i - width < 0:
            if i == 0 or i == width - 1:
                h[1, 0] = -0.5
                h[width, 0] = -0.5
                h[width - 2, width - 1] = -0.5
                h[2 * width - 1, width - 1] = -0.5
            else:
                h[i - 1, i] = -1 / 3
                h[i + 1, i] = -1 / 3
                h[i + width, i] = -1 / 3
        elif i + width > getGaussKernel - 1:
            if i == getGaussKernel- 1 or i == getGaussKernel - width:
                h[getGaussKernel - 2, getGaussKernel - 1] = -0.5
                h[getGaussKernel- width, getGaussKernel - 1] = -0.5
                h[getGaussKernel- width + 1, getGaussKernel- width] = -0.5
                h[getGaussKernel- 2 * width, getGaussKernel - width] = -0.5
            else:
                h[i - 1, i] = -1 / 3
                h[i + 1, i] = -1 / 3
                h[i - width, i] = -1 / 3
        elif i % width == 0:
            h[i - width, i] = -1 / 3
            h[i + width, i] = -1 / 3
            h[i + 1, i] = -1 / 3
        elif (i + 1) % width == 0:
            h[i - width, i] = -1 / 3
            h[i + width, i] = -1 / 3
            h[i - 1, i] = -1 / 3
    return h

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x

class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        # self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        B, N, C = x.shape
        # qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k = qk[0], qk[1]   # make torchscript happy (cannot use tensor as tuple)
        # v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        # q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """ TNT Block
    """

    def __init__(self, outer_dim, inner_dim, outer_num_heads, inner_num_heads, num_words, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            # Inner
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = Attention(
                inner_dim, inner_dim, num_heads=inner_num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.inner_norm2 = norm_layer(inner_dim)
            self.inner_mlp = Mlp(in_features=inner_dim, hidden_features=int(inner_dim * mlp_ratio),
                                 out_features=inner_dim, act_layer=act_layer, drop=drop)

            self.proj_norm1 = norm_layer(num_words * inner_dim)
            self.proj = nn.Linear(num_words * inner_dim, outer_dim, bias=False)
            self.proj_norm2 = norm_layer(outer_dim)
        # Outer
        self.outer_norm1 = norm_layer(outer_dim)
        self.outer_attn = Attention(
            outer_dim, outer_dim, num_heads=outer_num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.outer_norm2 = norm_layer(outer_dim)
        self.outer_mlp = Mlp(in_features=outer_dim, hidden_features=int(outer_dim * mlp_ratio),
                             out_features=outer_dim, act_layer=act_layer, drop=drop)
        # SE
        self.se = se
        self.se_layer = None
        if self.se > 0:
            self.se_layer = SE(outer_dim, 0.25)

    def forward(self, inner_tokens, outer_tokens):
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens)))  # B*N, k*k, c
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens)))  # B*N, k*k, c
            B, N, C = outer_tokens.size()
            outer_tokens[:, 1:] = outer_tokens[:, 1:] + self.proj_norm2(
                self.proj(self.proj_norm1(inner_tokens.reshape(B, N - 1, -1))))  # B, N, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        return inner_tokens, outer_tokens

class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """
    def __init__(self, img_size, patch_size, in_chans, inner_dim, inner_stride):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x)  # B, Ck2, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size)  # B*N, C, 16, 16
        x = self.proj(x)  # B*N, C, 8, 8
        x = x.reshape(B * self.num_patches, self.inner_dim, -1).transpose(1, 2)  # B*N, 8*8, C
        return x

class AE(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(3, 3, 6), stride=(1, 1, 2),
                                               padding=(1, 1, 0), bias=False), nn.BatchNorm3d(64),
                                     nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 4),
                                               stride=(1, 1, 2), padding=(1, 1, 0), bias=False),
                                     nn.ReLU(),
                                     nn.MaxPool3d(2, 2),  # [, 64, 48, 48]
                                     nn.BatchNorm3d(64),
                                     nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 5),
                                               stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(64),
                                     nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(1, 1, 3),
                                               stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(128),
                                     nn.Conv3d(in_channels=16, out_channels=8, kernel_size=(1, 1, 4),
                                               stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(128),
                                     nn.Conv3d(in_channels=8, out_channels=3, kernel_size=(1, 1, 3),
                                               stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.MaxPool3d(2, 2),
                                     nn.BatchNorm3d(256))
        self.decoder = nn.Sequential(nn.ConvTranspose3d(in_channels=8, out_channels=3, kernel_size=(1, 1, 3),
                                                        stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(128),
                                     nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(1, 1, 4),
                                                        stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(128),
                                     nn.ConvTranspose3d(in_channels=32, out_channels=16, kernel_size=(1, 1, 3),
                                                        stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(64),
                                     nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(3, 3, 5),
                                                        stride=(1, 1, 2), padding=(0, 0, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(32),
                                     nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 4),
                                                        stride=(1, 1, 2), padding=(1, 1, 0), bias=False),
                                     nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(3, 3, 4),
                                                        stride=(1, 1, 2), padding=(1, 1, 0), bias=False),
                                     nn.ReLU(),
                                     nn.BatchNorm3d(16),
                                     nn.ConvTranspose3d(in_channels=1, out_channels=128, kernel_size=(3, 3, 6),
                                                        stride=(1, 1, 2), padding=(1, 1, 0), bias=False),
                                     nn.ReLU())

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.0, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity
        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x

class TNT(nn.Module):
    """ TNT (Transformer in Transformer) for computer vision
    """
    def __init__(self, img_size, patch_size, in_chans, outer_dim, inner_dim, depth, outer_num_heads, inner_num_heads,
                 mlp_ratio, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=ContraNorm, inner_stride=4, se=0):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, inner_dim=inner_dim, inner_stride=inner_stride)
        self.num_patches = num_patches = self.patch_embed.num_patches
        num_words = self.patch_embed.num_words

        self.proj_norm1 = norm_layer(num_words * inner_dim)
        self.proj = nn.Linear(num_words * inner_dim, outer_dim)
        self.proj_norm2 = norm_layer(outer_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, outer_dim))
        self.outer_tokens = nn.Parameter(torch.randn(1, num_patches, outer_dim), requires_grad=False)
        self.outer_pos = nn.Parameter(torch.randn(1, num_patches + 1, outer_dim))
        self.inner_pos = nn.Parameter(torch.zeros(1, num_words, inner_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            if i in vanilla_idxs:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=-1, outer_num_heads=outer_num_heads, inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
            else:
                blocks.append(Block(
                    outer_dim=outer_dim, inner_dim=inner_dim, outer_num_heads=outer_num_heads,
                    inner_num_heads=inner_num_heads,
                    num_words=num_words, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, se=se))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(outer_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        inner_tokens = self.patch_embed(x) + self.inner_pos  # B*N, 8*8, C
        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        outer_tokens = outer_tokens + self.outer_pos
        outer_tokens = self.pos_drop(outer_tokens)
        for blk in self.blocks:
            inner_tokens, outer_tokens = blk(inner_tokens, outer_tokens)
            outer_tokens = self.norm(outer_tokens)
        return outer_tokens[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return x