import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import Pad
from functools import partial
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import argparse
import os
from timm.models.vision_transformer import Block

import src.models.transformer as transformer
import src.utils.utils as utils
import src.utils.losses as losses

global_path = "/home/ids/edabier/HSU"
# global_path = "/Users/edabier/Documents/Thèse/Thèse_Télécom"
# global_path = "/home/edabier/Documents/Thèse/benchmark"
sys.path.append(global_path)

sys.path.append(f"{global_path}/spectral_earth")
from spectral_earth.src.backbones import spec_vit
from spectral_earth.src.backbones import spec_resnet
from spectral_earth.src.backbones.spec_vit import SpecViTBase

sys.path.append(f"{global_path}/IEEE_TPAMI_SpectralGPT")
import models_mae_spectral

sys.path.append(f"{global_path}/HyperFree")
from HyperFree import build_HyperFree_vit_b, predictor
from HyperFree.modeling import image_encoder

sys.path.append(f"{global_path}/DOFA")
from wave_dynamic_layer import Dynamic_MLP_OFA

sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from models.model import SpatViT, SpecViT

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    print(f"{torch.cuda.is_available()}")
    dev = "cpu"

MODELS = ["SpectralEarth", "SpectralGPT", "DOFA", "HyperFree", "HyperSL", "HyperSIGMA"]

class HyperSIGMA_Unmix(torch.nn.Module):
    def __init__(self, patch_size, channels, seg_patches, NUM_TOKENS, embed_dim, num_em, scale):
        super(HyperSIGMA_Unmix, self).__init__()
        self.patch_size = patch_size
        self.spat_encoder = SpatViT(img_size=patch_size,
            in_chans=channels,
            use_checkpoint=True,
            patch_size=seg_patches,
            drop_path_rate=0.1, out_indices=[3, 5, 7, 11], embed_dim=768,
            depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., use_abs_pos_emb=False, n_points=8
        )
        self.spec_encoder = SpecViT(
            NUM_TOKENS=NUM_TOKENS,
            img_size=patch_size,
            in_chans=channels,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            use_checkpoint=False,
            use_abs_pos_emb=False,
            interval=3
        )

        self.conv_features1 = nn.Conv2d(embed_dim, NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec1 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features2 = nn.Conv2d(embed_dim, NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec2 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features3 = nn.Conv2d(embed_dim, NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec3 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.conv_features4 = nn.Conv2d(embed_dim, NUM_TOKENS, kernel_size=1, bias=False)
        self.fc_spec4 = nn.Sequential(
            nn.Linear(NUM_TOKENS, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, NUM_TOKENS, bias=False),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.conv1 =nn.Sequential(
            nn.Conv2d(NUM_TOKENS, NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),# nn.ReLU(),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(NUM_TOKENS, NUM_TOKENS, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(NUM_TOKENS, NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(NUM_TOKENS, NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )

        self.conv2_ = nn.Sequential(
            nn.Conv2d(NUM_TOKENS*2, NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv3_ = nn.Sequential(
            nn.Conv2d(NUM_TOKENS*2, NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.conv4_ = nn.Sequential(
            nn.Conv2d(NUM_TOKENS*2, NUM_TOKENS, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS),
            nn.Dropout(0.2),
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(NUM_TOKENS*4, NUM_TOKENS*2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(NUM_TOKENS*2),
            nn.Dropout(0.2),
            nn.Conv2d(NUM_TOKENS*2, NUM_TOKENS, kernel_size=(1, 1)) 
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(NUM_TOKENS, num_em, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(num_em),
            nn.Dropout(0.2),
        )

        self.sumtoone = Sum_to_one(scale)
        self.decoder = Decoder(c=num_em, B=channels)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
        x: (Variable) top feature map to be upsampled.
        y: (Variable) lateral feature map.
        Returns:
        (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.normal_(m.weight.data, mean=0.0, std=0.3)

    def forward_fusion(self, x):
        # x: (b, c, h, w)
        # ts:(b, c)
        # spat
        b, _, h, w = x.shape
        img_features = self.spat_encoder(x)

        img_fea = []
        ops = [self.conv_features1, self.conv_features2, self.conv_features3, self.conv_features4]
        for i in range(len(ops)):
            img_fea.append(ops[i](img_features[i+1]))

        spec_features = self.spec_encoder(x)
        spec_feature = spec_features[-1]
        spec_feature = self.pool(spec_feature).view(b, -1) # b, c

        spec_weights = []
        ops_ = [self.fc_spec1, self.fc_spec2, self.fc_spec3, self.fc_spec4]
        for i in range(len(ops_)):
            spec_weights.append((ops_[i](spec_feature)).view(b, -1, 1, 1))
        ss_feature = []
        ss_feature.append(x)
        for i in range(4):
            ss_feature.append((1 + spec_weights[i]) * img_fea[i])
        return ss_feature

    def getAbundances(self, x):
        H, W = x.shape[2], x.shape[3]
        x = self.forward_fusion(x) # x: list : 5
        p4 = self.conv1(x[4])
        p3 = self.conv2(x[3])
        p2 = self.conv3(x[2])
        p1 = self.conv4(x[1])
        p1 = torch.cat([p1,p2,p3,p4], dim=1)

        p1 = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=True)
        p1 = self.smooth(p1)
        x = self.conv5(p1)
        abunds = self.sumtoone(x)
        abunds = abunds
        return abunds

    def forward(self, patch):
        abunds = self.getAbundances(patch)
        output = self.decoder(abunds)
        return abunds,output

    def getEndmembers(self):
        endmembers = self.decoder.getEndmembers()
        if endmembers.shape[2] > 1:
            endmembers = np.squeeze(endmembers).mean(axis=2).mean(axis=2)
        else:
            endmembers = np.squeeze(endmembers)
        return endmembers

class OFAViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, drop_rate=0.,
                 embed_dim=1024, depth=24, num_heads=16, out_indices=[23], wv_planes=128, num_classes=45,
                 global_pool=True, mlp_ratio=4., norm_layer=nn.LayerNorm, use_cls=False, extend_cls=False):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)
        
        self.out_indices = out_indices
        self.use_cls = use_cls
        self.extend_cls = extend_cls

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, kernel_size=patch_size, embed_dim=embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, wave_list):
        # embed patches
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        x = x + self.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Use every block output of out_indices
            # The final features will be (N*embed_dim, num_patches+1)
            # Where N is the number of outputs in out_indices
            if i in self.out_indices:
                features.append(x[:, 1:, :].squeeze(0).T)

        if self.use_cls:
            features = x[:, 0, :]

        elif self.extend_cls:
            features = torch.cat(features, dim=0)
            cls = x[:, 0, :].T
            features = features + cls

        else:
            features = torch.cat(features, dim=0)

        return features

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x

def create_fm(fm_name, Y, c=None, n_features=1, patch_size=64, use_cls=False, extend_cls=False, path="/home/ids/edabier/HSU"):
    batch, B, H, _ = Y.shape
    device = Y.device

    if fm_name == "DOFA":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:new_H, :new_H]

        check_point = torch.load(f'{path}/DOFA/checkpoints/DOFA_ViT_base_e100.pth', map_location=device)
        if n_features == 1:
            out_indices = [11]
        elif n_features == 4:
            out_indices = [3,5,7,11]
        elif n_features == 9:
            out_indices = [i for i in range(3,12)]
        fm = OFAViT(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, out_indices=out_indices, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cls=use_cls, extend_cls=extend_cls)
        
        fm.load_state_dict(check_point, strict=False)

    elif fm_name == "HyperFree":
        n_patches = H//patch_size
        if n_patches%2 != 0:
            H = (n_patches-1)*patch_size + patch_size-1
        new_H = H
        Y = Y[:, :, :new_H, :new_H]

        checkpoint = torch.load(f"{path}/HyperFree/data/HyperFree-b.pth", map_location=device)
        fm = image_encoder.ImageEncoderViT(depth=12, embed_dim=768,
                img_size=H, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12, patch_size=patch_size, qkv_bias=True,
                use_rel_pos=True, global_attn_indexes=[5, 8, 11],
                merge_indexs = [3, 12], window_size=14, out_chans=256)
        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "HyperSIGMA":
        new_H = H

        embed_dim, seg_patches, NUM_TOKENS, scale = 768, 2, 64, 1
        fm = HyperSIGMA_Unmix(patch_size=patch_size, channels=B, seg_patches=seg_patches, NUM_TOKENS=NUM_TOKENS, embed_dim=embed_dim, num_em=c, scale=scale)

        spat_path = f"{path}/HyperSIGMA/HyperspectralUnmixing/data/spat-vit-base-ultra-checkpoint-1599.pth"
        spec_path = f"{path}/HyperSIGMA/HyperspectralUnmixing/data/spec-vit-base-ultra-checkpoint-1599.pth"
        Spat_pernet = torch.load(spat_path, map_location=torch.device('cpu'), weights_only=False)
        Spat_pernet = Spat_pernet['model']
        for k in list(Spat_pernet.keys()):
            if 'patch_embed.proj' in k:
                del Spat_pernet[k]
        for k in list(Spat_pernet.keys()):
            k_ = 'spat_encoder.' + k
            Spat_pernet[k_] = Spat_pernet.pop(k)

        Spec_pernet = torch.load(spec_path, map_location=torch.device('cpu'), weights_only=False)
        Spec_pernet = Spec_pernet['model']
        for k in list(Spec_pernet.keys()):
            if 'spec' in k:
                del Spec_pernet[k]
            if 'spat' in k:
                del Spec_pernet[k]
        for k in list(Spec_pernet.keys()):
            k_ = 'spec_encoder.' + k
            Spec_pernet[k_] = Spec_pernet.pop(k)

        model_params = fm.state_dict()
        same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        fm.load_state_dict(model_params)

        same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
        model_params.update(same_parsms)
        fm.load_state_dict(model_params)

    elif fm_name == "SpecViT":
        if H < 128:
            Y = F.interpolate(Y, size=(128,128))
            new_H = 128
        else:
            new_H = 128
        Y = Y[:,:,:new_H, :new_H]

        fm = spec_vit.SpecViTBase()
        checkpoint = torch.load(f"{path}/spectral_earth/data/data/spec_ViTb_mae.pth", map_location=dev)
        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "SpecRnDino":
        fm = spec_resnet.SpecResNet50(num_classes=0)
        checkpoint = torch.load(f"{path}/spectral_earth/data/data/spec_rn50_dino.pth", map_location=dev)
        fm.load_state_dict(checkpoint, strict=False)

    elif fm_name == "SpecRnMoco":
        fm = spec_resnet.SpecResNet50(num_classes=0)
        checkpoint = torch.load(f"{path}/spectral_earth/data/data/spec_rn50_moco.pth", map_location=dev)
        fm.load_state_dict(checkpoint, strict=False)
    
    else:
        print("Fm name is not known, use DOFA, HyperFree, HyperSIGMA, SpecViT, SpecRnDino or SpecRnMoco")
        return
    
    return fm, Y, new_H

def reshape_Y(fm_name, new_H, Y, A):
    if fm_name == "OFAViT": # DOFA
        if Y.shape[-1] < new_H:
            Y = F.interpolate(Y, size=(new_H,new_H))
        Y = Y[:,:,:new_H, :new_H]
        
        if A.shape[-1] < new_H:
            A = F.interpolate(A, size=(new_H,new_H))
        A = A[:,:,:new_H, :new_H]

    elif fm_name == "SpecViTBase":
        if Y.shape[-1] < new_H:
            Y = F.interpolate(Y, size=(new_H,new_H))
        Y = Y[:,:,:new_H, :new_H]
        
        if A.shape[-1] < new_H:
            A = F.interpolate(A, size=(new_H,new_H))
        A = A[:,:,:new_H, :new_H]
    
    else:
        return Y, A

    return Y, A

def extract_f(fm, Y, A, new_H, wavelengths, use_cls=False):
    fm_name = fm.__class__.__name__

    Y, A = reshape_Y(fm_name, new_H, Y, A)

    if fm_name == "OFAViT": # DOFA
        features = get_dofa_features(fm, Y, wavelengths)
        noise = torch.rand_like(features)

    elif fm_name == "ImageEncoderViT": # HyperFree
        features = get_hyperfree_features(fm, Y, wavelengths)
        noise = torch.rand_like(features)

    elif fm_name == "SpecViTBase":

        features = get_specvit_features(fm, Y, use_cls)
        noise = torch.rand_like(features)
    
    else:
        return
    
    return Y, A, features

def get_hypersigma_features(fm, Y, patch_size=64):
    """
    Extracts features from the input HSI (features must be extracted patch by patch, 
    processed independently and stitched back together only at pixel level and not feature level)
    """
    batch, B, H, _ = Y.shape

    if H > patch_size:
        Y = Y[:, :, :patch_size, :patch_size]
        print("Cutted end of Y to forward only (patch, patch) to hypersigma")
    x = fm.forward_fusion(Y)
    p4 = fm.conv1(x[4])
    p3 = fm.conv2(x[3])
    p2 = fm.conv3(x[2])
    p1 = fm.conv4(x[1])
    features = torch.cat([p1,p2,p3,p4], dim=1)

    return features

def unmix_full_image_hypersigma(Y, unmixer, fm, c, patch_size=64, use_bn=True):

    batch, B, H, W = Y.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    Y_pad = F.pad(Y, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = Y_pad.shape

    # extract patches (batch, B*p*p, N), N = number of patches per image
    patches = F.unfold(
        Y_pad,
        kernel_size=patch_size,
        stride=patch_size
    )

    N = patches.shape[-1]

    # reshape to (batch*N, B, p, p)
    patches = patches.transpose(1, 2)
    patches = patches.reshape(batch * N, B, patch_size, patch_size)

    # extract all patches' features in parallel (batch)
    features = get_hypersigma_features(
        fm, patches, patch_size=patch_size
    )

    # unmix all patches at once
    noise = torch.rand_like(features)
    if use_bn:
        _, A_hat, Y_hat = unmixer(features) # A_hat: (batch*N, c, p, p), Y_hat: (batch*N, B, p, p)
    
    else:
        _, A_hat, Y_hat = unmixer.forward_nobn(features)

    # reshape back
    A_hat = A_hat.reshape(batch, N, c, patch_size, patch_size)
    Y_hat = Y_hat.reshape(batch, N, B, patch_size, patch_size)

    # reorder to (batch, c*p*p, N)
    A_hat = A_hat.permute(0, 2, 3, 4, 1).reshape(batch, c * patch_size * patch_size, N)
    Y_hat = Y_hat.permute(0, 2, 3, 4, 1).reshape(batch, B * patch_size * patch_size, N)

    # fold back
    A_full = F.fold(
        A_hat,
        output_size=(Hp, Wp),
        kernel_size=patch_size,
        stride=patch_size
    )

    Y_full = F.fold(
        Y_hat,
        output_size=(Hp, Wp),
        kernel_size=patch_size,
        stride=patch_size
    )

    # crop back
    A_full = A_full[:, :, :H, :W]
    Y_full = Y_full[:, :, :H, :W]

    E_hat = unmixer.get_endmembers()

    return E_hat, A_full, Y_full

def get_specvit_features(fm, Y, use_cls=False):
    features = fm(Y)

    if use_cls:
        features = features[:, 0, :]
    else:
        features = features[:, 1:, :]
        features = features[0].T

    return features

def get_hyperfree_features(fm, Y, wavelengths, GSD=torch.tensor([1.0])):
    """
    Extracts features from input hsi
    For HyperFree, the number of patches must be even (as patch merging will divide it by 2)
    So we make sure to cut the image to ensure even patch number

    Args:
        fm: The hyperfree model to perform inference
        Y: The input hsi on which to extract features
        wavelengths: The list of wavelengths of the input hsi
        GSD: Ground Sampling Distance, the spatial resolution of the hsi (m/pixel)
    """
    if GSD == None:
        GSD = 0.456
        ratio = 1024 / Y.shape[2]
        GSD = GSD / ratio
        GSD = torch.tensor([GSD])
    
    n_patches = Y.shape[2]//fm.patch_size
    if n_patches%2 != 0:
        H = (n_patches-1)*fm.patch_size + fm.patch_size-1
        Y = Y[:, :, :H, :H]

    features = fm(Y, input_wavelength=wavelengths, GSD=GSD)[-1]
    features = features.reshape(256, features.shape[2]**2).squeeze(0)
    return features

def get_dofa_features(fm, Y, wavelengths):
    features = fm.forward_features(Y, wavelengths)
    return features

def features_comparison(model_name, Y_gt, Y_hat, wavelengths=None, c=None):
    """
    computes the L2 norm between the features extracted from Y_gt 
    and those extraced from Y_hat by some RSFM
    """
    mse = nn.MSELoss()
    sad = losses.SADLoss()

    if model_name == "DOFA":
        features = get_dofa_features(Y_gt, wavelengths)
        features_hat = get_dofa_features(Y_hat, wavelengths)
    elif model_name == "HyperSIGMA":
        features = get_hypersigma_features(Y_gt, c)
        features_hat = get_hypersigma_features(Y_hat, c)

    sim = sad(features, features_hat)
    return sim

class Weight_constraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.clamp_(min=0)

class Sum_to_one(nn.Module):
    def __init__(self, scale=1):
        super(Sum_to_one, self).__init__()
        self.scale = scale
    def forward(self, x):
        x = F.softmax(self.scale * x, dim=1)
        return x

class Decoder(nn.Module):
    def __init__(self, c, B, kernel_size=1, is_cnnaeu=False):
        super(Decoder, self).__init__()
        self.B = B
        self.c = c
        self.is_cnnaeu = is_cnnaeu

        if self.is_cnnaeu:
            self.decoder = nn.Conv2d(self.c, self.B, kernel_size=11, padding=5, padding_mode="reflect", bias=False)

        else:
            padding = kernel_size //2
            self.decoder = nn.Conv2d(in_channels=c, out_channels=B,
                                    kernel_size=kernel_size,stride=1,
                                    padding=padding, bias=False)
            self.relu = nn.ReLU()

    def forward(self, code):

        if self.is_cnnaeu:
            code = self.decoder(code)
        
        else:
            code = self.relu(self.decoder(code))
        
        return code

    def get_endmembers(self):
        if self.is_cnnaeu:
            e_hat = self.decoder.weight.detach().mean((2, 3))
            e_hat = e_hat.reshape(self.B, self.c)
            return e_hat    
        
        else:
            return self.decoder.weight.data.squeeze([2, 3])

class Unmixing_from_features(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, n_features=1, use_cls=False, hypersig=False, is_cnnaeu=False):
        """
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features
            n_features (int): The size of the list of features in the case of several extracted features

        """
        super(Unmixing_from_features, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H
        self.n_features = n_features
        self.use_cls = use_cls
        self.hypersig = hypersig

        # Upsampling features
        if self.use_cls:
            self.upsample = nn.Sequential(
                nn.Linear(int(D/c), self.H**2)
            )
        elif not self.hypersig:
            self.upsample = nn.Sequential(
                nn.Linear(self.n_features*(self.alpha**2), self.H**2)
            )
        else:
            pass

        # Upsampled features to abundances
        if self.hypersig:
            self.abundance_estimator = nn.Sequential(
                nn.Conv2d(D*4, D*2, kernel_size=(3, 3), padding=1, bias=False),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(D*2),
                nn.Dropout(0.2),
                nn.Conv2d(D*2, D, kernel_size=(1, 1)),
                nn.Conv2d(D, c, kernel_size=1, bias=False),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(c),
                nn.Dropout(0.2),
            )
        else:
            if self.use_cls:
                self.smooth = nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.Softmax(dim=1),
                )

            else:
                self.abundance_estimator = nn.Sequential(
                    nn.Conv2d(D, c, kernel_size=1, bias=False),
                    nn.LeakyReLU(0.02),
                    nn.BatchNorm2d(c),
                    nn.Dropout(0.2)
                )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c, is_cnnaeu=is_cnnaeu)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.35, W_tv_e=0.1, W_tv_a=0, W_mse=0, W_e=0, hypersigma=False):
        sad = losses.SADLoss()
        tv = losses.TVLoss(reduction="mean")
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        if hypersigma:
            loss_mse = W_mse * losses.hypersigma_mse(Y_gt, Y_hat)
        else:
            loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)
        
        # Enforce Sum of all wavelength to be 1 for each endmember:
        # Minimize 1 - sum(E[:,i]) for i in c
        loss_norm_e = W_e * torch.norm(torch.ones(E_hat.shape[1]) - torch.sum(E_hat, dim=0))

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        # TV on abundances (sum of difference between consecutive horizontal pixels + vertical pixels)
        loss_tv_a = W_tv_a * tv(A_hat)

        loss = loss_sad + loss_ab + loss_tv_e + loss_tv_a + loss_mse + loss_norm_e

        return loss, loss_sad, loss_ab, loss_tv_e, loss_tv_a, loss_mse, loss_norm_e

    def freeze_decoder(self):
        """
        Normalizes the weights of the decoder and freezes it
        """
        state_dict = self.decoder.state_dict()
        E_hat = self.decoder.get_endmembers()
        E_hat = utils.normalize(E_hat, is_endmember=True)
        state_dict["decoder.weight"][:, :, 0, 0] = E_hat
        self.decoder.load_state_dict(state_dict)

        for param in self.decoder.parameters():
            param.requires_grad = False

    @staticmethod
    def supervised_loss(Y_gt, Y_hat, A_gt, A_hat, E_gt, E_hat):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')

        loss_re = sad(Y_gt, Y_hat)
        loss_mse = mse(A_gt, A_hat)/(torch.norm(A_gt)**2)
        loss_sad = sad(E_gt, E_hat)
        loss = loss_re + loss_mse + loss_sad

        return loss, loss_re, loss_sad, loss_mse
    
    def get_abundances(self, features):

        if self.use_cls:
            features_2d = features.reshape(self.c, int(self.D/self.c))
            features_up = self.upsample(features_2d)
            A_hat = features_up.reshape(1, self.c, self.H, self.H)
            A_hat = self.smooth(A_hat)           

        elif self.hypersig:
            if features.dim() == 2:
                features = features.reshape(1, 4*self.D, self.alpha, self.alpha)
            features_up = F.interpolate(features, size=(self.H, self.H), mode='bilinear', align_corners=True)
            features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
            A_hat = self.abundance_estimator(features_up)

        else:
            features = features.reshape(self.D, self.n_features*self.alpha*self.alpha)
            features_up = self.upsample(features)
            features_up = features_up.view(
                1, self.D, self.H, self.H
            )
            features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
            A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class CNNAEU_with_decoder(nn.Module):
    def __init__(self, B, c, decoder=None, E_init=None, freeze_E=True):
        super().__init__()

        self.B = B
        self.c = c
        self.encoder = nn.Sequential(
            nn.Conv2d(self.B, 48, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.c, kernel_size=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.c),
            nn.Dropout2d(p=0.2),
        )
        if decoder != None:
            self.decoder = decoder
        else:
            self.decoder = Decoder(self.c, self.B)
            if E_init != None:
                state_dict = self.decoder.state_dict()
                state_dict["decoder.weight"] = E_init.unsqueeze(-1).unsqueeze(-1)
                self.decoder.load_state_dict(state_dict)

        if freeze_E:
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        sad = losses.SADLoss()
        return sad(Y_gt, Y_hat)
    
    def forward(self, x):
        # Input shape (batch, B, N)
        
        if x.dim() < 3:
            x = x.unsqueeze(0) # Add a batch dimension for inference
        
        batch, B, N = x.shape
        x = utils.oneD_to_2d(x)
        
        code = self.encoder(x)
        
        abund = F.softmax(code, dim=1)
        a_hat = abund.reshape(batch, self.c, N)
        
        x_hat = self.decoder(abund)
        x_hat = x_hat.reshape(batch, B, N)
        
        e_hat = self.decoder.get_endmembers()
        
        return e_hat, a_hat, x_hat