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

import src.utils.utils as utils

global_path = "/home/ids/edabier/HSU"
sys.path.append(global_path)

from spectral_earth.src.backbones.spec_vit import SpecViTBase

sys.path.append(f"{global_path}/IEEE_TPAMI_SpectralGPT")
import models_mae_spectral

sys.path.append(f"{global_path}/HyperFree")
from HyperFree import build_HyperFree_vit_b, predictor
from HyperFree.modeling import image_encoder

sys.path.append(f"{global_path}/DOFA")
from dofa_v1 import vit_base_patch16

sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from models.model import SpatViT, SpecViT

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    print(f"{torch.cuda.is_available()}")
    dev = "cpu"

MODELS = ["SpectralEarth", "SpectralGPT", "DOFA", "HyperFree", "HyperSL", "HyperSIGMA"]

# def create_fm(fm_name, Y, c):
#     batch, B, H, _ = Y.shape


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

def get_hypersigma_features(Y, c, patch_size=64):
    """
    Extracts features from the input HSI (features must be extracted patch by patch, 
    processed independently and stitched back together only at pixel level and not feature level)
    """
    batch, B, H, _ = Y.shape
        
    embed_dim, seg_patches, NUM_TOKENS, scale = 768, 2, 64, 1
    n_patches = H//patch_size
    hypersigma = HyperSIGMA_Unmix(patch_size=patch_size, channels=B, seg_patches=seg_patches, NUM_TOKENS=NUM_TOKENS, embed_dim=embed_dim, num_em=c, scale=scale)

    spat_path = "/home/ids/edabier/HSU/HyperSIGMA/HyperspectralUnmixing/data/spat-vit-base-ultra-checkpoint-1599.pth"
    spec_path = "/home/ids/edabier/HSU/HyperSIGMA/HyperspectralUnmixing/data/spec-vit-base-ultra-checkpoint-1599.pth"
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

    model_params = hypersigma.state_dict()
    same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
    model_params.update(same_parsms)
    hypersigma.load_state_dict(model_params)

    same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
    model_params.update(same_parsms)
    hypersigma.load_state_dict(model_params)

    if H > patch_size:
        Y = Y[:, :, :patch_size, :patch_size]
        print("Cutted end of Y to forward only (patch, patch) to hypersigma")
    x = hypersigma.forward_fusion(Y)
    p4 = hypersigma.conv1(x[4])
    p3 = hypersigma.conv2(x[3])
    p2 = hypersigma.conv3(x[2])
    p1 = hypersigma.conv4(x[1])
    features = torch.cat([p1,p2,p3,p4], dim=1)

    return features

# def unmix_full_image_hypersigma(Y, model, c, patch_size=64, overlap=0):
#     """
#     Patch-based hyperspectral unmixing using HyperSIGMA.
#     Uses patch_size and optional overlap to tile the input image,
#     extracts features with rsfm.get_hypersigma_features,
#     then runs the unmixing model on each patch,
#     and reconstructs the full abundance & reconstruction maps.
#     """

#     device = Y.device
#     batch, B, H, W = Y.shape

#     # stride determines patch shifting
#     stride = patch_size - overlap

#     # Pad the image so that it is divisible by stride exactly
#     pad_h = H - patch_size
#     pad_w = W - patch_size

#     Y_pad = F.pad(Y, (0, pad_w, 0, pad_h), mode="reflect")

#     _, _, Hp, Wp = Y_pad.shape

#     A_full = torch.zeros(batch, c, Hp, Wp, device=device, dtype=Y.dtype)
#     Y_full = torch.zeros(batch, B, Hp, Wp, device=device, dtype=Y.dtype)
#     weight = torch.zeros(1, 1, Hp, Wp, device=device, dtype=Y.dtype)

#     for i in range(0, Hp - patch_size):
#         for j in range(0, Wp - patch_size):

#             Y_patch = Y_pad[:, :, i:i+patch_size, j:j+patch_size]

#             features = get_hypersigma_features(Y_patch, c, patch_size=patch_size)
#             # noise = torch.rand_like(features)
#             E_hat, A_hat, Y_hat = model(features)

#             # accumulate (WITH grad)
#             A_full[:, :, i:i+patch_size, j:j+patch_size] += A_hat
#             Y_full[:, :, i:i+patch_size, j:j+patch_size] += Y_hat
#             weight[:, :, i:i+patch_size, j:j+patch_size] += 1.0

#     # average overlaps (still differentiable)
#     A_full = A_full / weight
#     Y_full = Y_full / weight

#     # crop back to original size
#     A_full = A_full[:, :, :H, :W]
#     Y_full = Y_full[:, :, :H, :W]

#     return E_hat, A_full, Y_full

def unmix_full_image_hypersigma(Y, model, c, patch_size=64):
    device = Y.device
    batch, B, H, W = Y.shape

    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    Y_pad = F.pad(Y, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, Hp, Wp = Y_pad.shape

    A_full = torch.zeros(batch, c, Hp, Wp, device=device, dtype=Y.dtype)
    Y_full = torch.zeros(batch, B, Hp, Wp, device=device, dtype=Y.dtype)

    for i in range(0, Hp ,patch_size):
        for j in range(0, Wp, patch_size):

            Y_patch = Y_pad[:, :, i:i+patch_size, j:j+patch_size]
            features = get_hypersigma_features(Y_patch, c, patch_size=patch_size)
            _, A_hat, Y_hat = model(features)

            A_full[:, :, i:i+patch_size, j:j+patch_size] = A_hat
            Y_full[:, :, i:i+patch_size, j:j+patch_size] = Y_hat

    A_full = A_full[:, :, :H, :W]
    Y_full = Y_full[:, :, :H, :W]

    E_hat = model.get_endmembers()

    return E_hat, A_full, Y_full

def get_hyperfree_features(Y, wavelengths):
    device = Y.device
    batch, B, H, _ = Y.shape
    checkpoint = torch.load("/home/ids/edabier/HSU/HyperFree/data/HyperFree-b.pth", map_location=device)
    hyperfree = image_encoder.ImageEncoderViT(depth=12, embed_dim=768,
            img_size=H, mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12, patch_size=16, qkv_bias=True,
            use_rel_pos=True, global_attn_indexes=[5, 8, 11],
            merge_indexs = [3, 12], window_size=14, out_chans=256)
    hyperfree.load_state_dict(checkpoint, strict=False)
    features = hyperfree(Y, input_wavelength=wavelengths)[-1]
    features = features.reshape(256, features.shape[2]**2)
    return features

def get_dofa_features(Y, wavelengths, n_features=1):
    device = Y.device
    check_point = torch.load('/home/ids/edabier/HSU/DOFA/checkpoints/DOFA_ViT_base_e100.pth', map_location=device)
    dofa = vit_base_patch16(n_features=n_features)
    dofa.load_state_dict(check_point, strict=False)
    features = dofa.forward_features(Y, wavelengths)
    return features

class Foundation_model(nn.Module):
    def __init__(self, model_name, patch_size=None, im_size=None, channels=None, n_em=None, wavelengths=None):
        """
        Instantiate the provided foundation model
        """
        super(FoundationModel, self).__init__()

        if model_name not in MODELS:
            raise ValueError("The provided model_name does not correspond to any of [SpectralEarth, SpectralGPT, DOFA, HyperFree, HyperSL, HyperSIGMA]")

        self.model_name = model_name
        self.patch_size = patch_size
        self.channels = channels
        self.n_em = n_em

        if model_name == "SpectralEarth":
            model = SpecViTBase()
            model.vit_core.head = torch.nn.Identity()
            state_dict = torch.load(f"{global_path}/spectral_earth/data/data/spec_ViTb_mae.pth")
            model.load_state_dict(state_dict, strict=False)

            self.model = model
            self.im_size = 128
            # self.unmixer = Unmixing_from_features(self.n_em, self.im_size, self.channels)

        # elif model_name == "SpectralGPT":
            # assert channels is not None, "The number of channels must be specified for SpectralGPT"
            # state_dict = torch.load(f"{global_path}/IEEE_TPAMI_SpectralGPT/data/SpectralGPT+.pth", map_location=dev, weights_only=False)["model"]
            # self.im_size = 128
            # spectral_gpt = models_mae_spectral.mae_vit_base_patch8_128(num_frames=channels, pred_t_dim=channels)
            # # state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            # # spectral_gpt.load_state_dict(state_dict, strict=False)
            # encoder_keys = [k for k in state_dict.keys() if k.startswith('patch_embed') or k.startswith('blocks') or k.startswith('norm')]
            # encoder_state_dict = {k: state_dict[k] for k in encoder_keys}
            # spectral_gpt.load_state_dict(encoder_state_dict, strict=False)
                        
            # self.model = spectral_gpt
            
        elif model_name == "DOFA":
            assert self.channels is not None, "channels must be set"
            assert self.n_em is not None, "n_em must be set"
            assert wavelengths is not None, "wavelengths must be set"

            state_dict = torch.load("/home/ids/edabier/HSU/DOFA/checkpoints/DOFA_ViT_base_e100.pth", map_location=dev)
            model = vit_base_patch16()
            model.load_state_dict(state_dict, strict=False)

            self.im_size = 224
            self.wavelengths = wavelengths
            self.model = model
            # self.unmixer = Unmixing_from_features(self.n_em, self.im_size, self.channels)

        elif model_name == "HyperFree":
            assert self.patch_size is not None, "Patch_size must be set"
            assert wavelengths is not None, "wavelengths must be set"
            assert im_size is not None, "im_size must be set"

            self.im_size = im_size
            self.wavelengths = wavelengths
            pred = build_HyperFree_vit_b(checkpoint="/home/ids/edabier/HSU/HyperFree/data/HyperFree-b.pth", image_size=im_size, vit_patch_size=patch_size)
            self.model = predictor.HyperFree_Predictor(pred)

        elif model_name == "HyperSL":
            pass

        elif model_name == "HyperSIGMA":
            assert self.channels is not None, "The number of channels must be specified for HyperSIGMA"
            assert self.n_em is not None, "The number of endmembers must be specified for HyperSIGMA"

            parser = argparse.ArgumentParser()
            parser.add_argument('--patch_size', default=64)
            parser.add_argument('--seg_patches', default=2)
            parser.add_argument('--embed_dim', default=768)
            parser.add_argument('--NUM_TOKENS', default=64)
            parser.add_argument('--channels', default=channels)
            parser.add_argument('--num_em', default=n_em)
            parser.add_argument('--kernel', default=1)
            parser.add_argument('--scale', default=1, type=float)
            args, _ = parser.parse_known_args()

            model = HyperSIGMA_Unmix(args)

            Spat_pernet = torch.load("/home/ids/edabier/HSU/HyperSIGMA/HyperspectralUnmixing/data/spat-vit-base-ultra-checkpoint-1599.pth", map_location=torch.device('cpu'), weights_only=False)
            Spat_pernet = Spat_pernet['model']
            for k in list(Spat_pernet.keys()):
                if 'patch_embed.proj' in k:
                    del Spat_pernet[k]
            for k in list(Spat_pernet.keys()):
                k_ = 'spat_encoder.' + k
                Spat_pernet[k_] = Spat_pernet.pop(k)

            Spec_pernet = torch.load("/home/ids/edabier/HSU/HyperSIGMA/HyperspectralUnmixing/data/spec-vit-base-ultra-checkpoint-1599.pth", map_location=torch.device('cpu'), weights_only=False)
            Spec_pernet = Spec_pernet['model']
            for k in list(Spec_pernet.keys()):
                if 'spec' in k:
                    del Spec_pernet[k]
                if 'spat' in k:
                    del Spec_pernet[k]
            for k in list(Spec_pernet.keys()):
                k_ = 'spec_encoder.' + k
                Spec_pernet[k_] = Spec_pernet.pop(k)

            model_params = model.state_dict()
            same_parsms = {k: v for k, v in Spat_pernet.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)

            same_parsms = {k: v for k, v in Spec_pernet.items() if k in model_params.keys()}
            model_params.update(same_parsms)
            model.load_state_dict(model_params)
            self.model = model

    def get_features(self, Y):
        """
        Forwards the input HSI to the model's encoder to get features

        Args:
            Y: input HSI tensor of shape (batch, B, H, W)
            for HyperFree, H and W must be multiples of patch_size and values must be normalized (/max(Y))
        """
        if Y.dim() == 4:
            batch, B, H, W = Y.shape
        elif Y.dim() == 3:
            B, H, W = Y.shape
            Y = Y.unsqueeze(0)
        else:
            print("Input HSI must be of shape (B, H, W) or (batch, B, H, W)")
            return
        
        if H > self.im_size:
            print(f"Input HSI larger than expected size ({self.im_size}), cutting it to match")
            Y = Y[:, :, :self.im_size, :self.im_size]
        elif H < self.im_size:
            # TO DO
            print(f"Input HSI smaller than expected size ({self.im_size}), padding to match")

        if self.model_name == "HyperFree":
            assert self.wavelengths is not None, "HyperFree needs wavelengths list for spectral embedding"
            GSD = 0.456
            ratio = 1024 / (max(Y.shape[2], Y.shape[3]))
            GSD = GSD / ratio
            GSD = torch.tensor([GSD])

            input_im = self.model.transform.apply_image_torch(Y)
            self.model.set_torch_image(input_im, original_image_size=(Y.shape[1], Y.shape[2]), spectral_lengths=self.wavelengths, GSD=GSD)

            return self.model.features
        
        elif self.model_name == "DOFA":
            assert self.wavelengths is not None, "DOFA needs wavelengths list for the positional encoding"
            return self.model.forward_features(Y, wave_list=self.wavelengths) 
        
        elif self.model_name == "SpectralEarth":
            return self.model(Y)
        
        elif self.model_name == "HyperSIGMA":
            features = self.model.forward_fusion(Y)
            return features
        
        else:
            pass
    
    def get_abundances(self, F, Y):
        if self.model_name == "SpectralEarth" or self.model_name == "DOFA":

            if Y.shape[-1] > self.im_size:
                Y = Y[:, :self.im_size, :self.im_size]

            A_hat = self.encoder(F, Y)
            return A_hat

    def forward(self, Y):
        """
        Unmix the input HSI by extracting features using the rsfm, and using them to obtain abundances and endmembers

        Args:
            Y: input HSI tensor of shape (batch, B, H, W) or (B, H, W)
            c: the number of endmembers to unmix
        """
        if Y.dim() == 4:
            batch, B, H, W = Y.shape
            Y = Y.squeeze(0)
        elif Y.dim() == 3:
            B, H, W = Y.shape
            # Y = Y.unsqueeze(0)
        else:
            print("Input HSI must be of shape (B, H, W) or (batch, B, H, W)")
            return 

        F = self.get_features(Y)
        A = self.get_abundances(F, Y)
        Y_hat = self.decoder(A)
        E = self.decoder.get_endmembers()

        return E, A, Y_hat

    def get_adapter_size(self):
        print("Adapter has", sum(p.numel() for p in self.encoder.parameters() if p.requires_grad + p.numel() for p in self.decoder.parameters() if p.requires_grad)/1e3, "k params")

def features_comparison(model_name, Y_gt, Y_hat, wavelengths=None, c=None):
    """
    computes the L2 norm between the features extracted from Y_gt 
    and those extraced from Y_hat by some RSFM
    """
    mse = nn.MSELoss()
    sad = utils.SADLoss()

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
    def __init__(self, c, B, kernel_size=1):
        super(Decoder, self).__init__()
        padding = kernel_size //2
        self.decoder = nn.Conv2d(in_channels=c, out_channels=B,
                                kernel_size=kernel_size,stride=1,
                                padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, code):
        code = self.relu(self.decoder(code))
        return code

    def get_endmembers(self):
        return self.decoder.weight.data.squeeze([2, 3])
    
class Upsample_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.conv_transpose(x)

def upsample_features(model_name, patch_size, Y, wavelengths=None):
    """
    Author: Antoine Domingues
    Extracts features from the input hsi Y and a padded version to create upsampled
    features by averaging the overlapping features
    
    Args:
        model_name: Which FM to use to extract features
        patch_size: The patch size used by the model
        Y: The input hsi from which to extract the features (must be of shape (batch, B, H, W) or (B, H, W))
    """

    if Y.dim() < 4:
        Y = Y.unsqueeze(0)

    padding_size = patch_size//2
    padding = Pad(padding_size)
    input_padded_img = padding(Y)

    # Extract the features of the two views
    H = Y.shape[2]

    if model_name == "DOFA":
        assert wavelengths != None, "Wavelengths list must be set for DOFA features extraction"
        extracted_features = get_dofa_features(Y, wavelengths).reshape(1, H//patch_size, H//patch_size, -1) # (1, 14, 14, 768)
        extracted_features_shifted = get_dofa_features(input_padded_img, wavelengths).reshape(1, H//patch_size + 1, H//patch_size + 1, -1) # (1, ?, ?, 768)

    # Might be more difficult as DOFA needs input shape to be exactly 224 and thus cannot take padded input

    extracted_features = extracted_features.permute(0, 3, 1, 2)
    extracted_features_shifted = extracted_features_shifted.permute(0, 3, 1, 2)

    # Duplicate the features via nearest neighbor interpolation to match the final resolution
    features_up = torch.nn.functional.interpolate(extracted_features, scale_factor=2, mode='nearest')
    features_up_shifted = torch.nn.functional.interpolate(extracted_features_shifted, scale_factor=2, mode='nearest')

    # Take only the center features
    features_up_shifted = features_up_shifted[:, :, 1:-1, 1:-1]

    # Gather the features to perform the average pixel-wise
    all_features = torch.cat((features_up, features_up_shifted), dim=0).permute(0, 2, 3, 1) # (2, 128, 128, 1024)
    features_up = all_features.mean(dim=0, keepdim=True)
    features_flat_up = features_up.flatten(1, 2)
    return features_flat_up

class Unmixing_from_features(nn.Module):
    def __init__(self, D, p, B, c, use_N=False, H=224, n_features=1, use_cls=False, hypersig=False, upsample_twice=False):
        super(Unmixing_from_features, self).__init__()
        self.D = D
        self.p = p
        self.B = B
        self.c = c
        self.H = H
        self.use_N = use_N
        self.n_features = n_features
        self.use_cls = use_cls
        self.hypersig = hypersig
        self.upsample_twice = upsample_twice

        if self.use_cls:
            if self.use_N:
                self.upsample = nn.Sequential(
                    nn.Linear(int(D/c), self.p**2)
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Linear(int(D/c), self.H**2)
                )
        elif self.hypersig:
            if self.use_N:
                self.upsample = nn.Sequential(
                    nn.Linear((p//2)**2, self.p**2)
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Linear(p**2, self.H**2)
                )        
        else:
            if self.use_N:
                self.upsample = nn.Sequential(
                    nn.Linear(p**2, self.p**2)
                )
            else:
                self.upsample = nn.Sequential(
                    nn.Linear(self.n_features*(p**2), self.H**2)
                )

        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        if self.hypersig:
            self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D*4, D*2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(D*2),
            nn.Dropout(0.2),
            nn.Conv2d(D*2,  D, kernel_size=(1, 1)),
            nn.Conv2d(D, c, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
            )
        else:
            self.abundance_estimator = nn.Sequential(
                nn.Conv2d(D, c, kernel_size=1),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(c),
                nn.Dropout(0.2)
            )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.35, W_tv=0.1, W_mse=0):
        sad = utils.SADLoss()
        mse = nn.MSELoss(reduction='mean')
        
        loss_sad = sad(Y_gt, Y_hat)
        loss_ab = torch.sqrt(A_hat).mean()
        loss_tv = (torch.abs(E_hat[:, 1:] - E_hat[:, :(-1)]).sum())
        loss_mse = mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        loss = W_sad * loss_sad + W_ab * loss_ab + W_tv * loss_tv + W_mse * loss_mse 

        return loss
    
    def get_abundances(self, features):

        if self.use_cls:
            features_2d = features.reshape(self.c, int(self.D/self.c))
            features_up = self.upsample(features_2d)
            if self.use_N:
                A_hat = features_up.reshape(1, self.c, self.p, self.p)
            else:
                A_hat = features_up.reshape(1, self.c, self.H, self.H)            

        elif self.hypersig:
            features = features.reshape(1, 4*self.D, (self.p//2)**2)
            features_up = self.upsample(features)
            if self.use_N:
                features_up = features_up.view(
                    1, 4*self.D, self.p, self.p
                )
            else:
                features_up = features_up.view(
                    1, 4*self.D, self.H, self.H
                )

            noise = torch.rand_like(features_up)
            A_hat = self.abundance_estimator(features_up)

        else:
            features_2d = features.view(
                self.D, int(self.n_features**0.5)*self.p, int(self.n_features**0.5)*self.p
            )

            features = features.reshape(self.D, self.n_features*self.p*self.p)
            features_up = self.upsample(features)
            if self.use_N:
                features_up = features_up.view(
                    1, self.D, int(self.N**0.5), int(self.N**0.5)
                )
            else:
                features_up = features_up.view(
                    1, self.D, self.H, self.H
                )

            A_hat = self.abundance_estimator(features_up)

        A_hat = self.smooth(A_hat)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat

# class Unmixing_from_features(nn.Module):
#     def __init__(self, D, B, c):
#         super(Unmixing_from_features, self).__init__()
#         self.D = D
#         self.B = B
#         self.c = c

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(D*4, D*2, kernel_size=(3, 3), padding=1),
#             nn.LeakyReLU(0.02),
#             nn.BatchNorm2d(D*2),
#             nn.Dropout(0.2),
#             nn.Conv2d(D*2,  D, kernel_size=(1, 1)) )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(D, c, kernel_size=1),
#             nn.LeakyReLU(0.02),
#             nn.BatchNorm2d(c),
#             nn.Dropout(0.2),
#         )

#         self.sum_to_one = Sum_to_one()
#         self.decoder = Decoder(B=B, c=c)
    
#     def abundances_from_features(self, features):

#         self.p = self.p = int(features.shape[1] ** 0.5)
#         features_2d = features.view(
#             1, 4*self.D, self.p, self.p
#         )
#         features_up = F.interpolate(
#             features_2d,
#             scale_factor=16,
#             mode="bilinear",
#             align_corners=True
#         )

#         patch_D = self.conv1(features_up)
#         A_hat   = self.conv2(patch_D)
#         A_hat   = self.sum_to_one(A_hat)

#         return A_hat
    
#     def forward(self, features):
#         A_hat = self.abundances_from_features(features)
#         Y_hat = self.decoder(A_hat)
#         E_hat = self.decoder.get_endmembers()

#         return E_hat, A_hat, Y_hat

# class Abundances_from_features(nn.Module):
#     """
#     A lightweight "encoder" using ViT 1D feature vector and input HSI to obtain abundances estimates
#     Assumes a square image (H=W)
    
#     Args:
#         c (int): the number of endmembers to extract
#         H (int): the shape of the input HSI
#         B (int): the number of spectral bands in the input HSI
#     """
#     def __init__(self, c, H, B):
#         super().__init__()
#         self.H = H
#         self.c = c
#         self.B = B

#         # Step 1: Upsample from 16 to 64
#         self.upsample1 = nn.ConvTranspose2d(
#             in_channels=3,
#             out_channels=3,
#             kernel_size=8,
#             stride=4,
#             padding=2,
#         )
#         # Step 2: Upsample from 64 to H
#         stride2 = H // 64
#         kernel_size2 = 2 * stride2
#         padding2 = kernel_size2 // 2 - 1
#         self.upsample2 = nn.ConvTranspose2d(
#             in_channels=3,
#             out_channels=3,
#             kernel_size=kernel_size2,
#             stride=stride2,
#             padding=padding2,
#         )

#         # Reduce y channels: (B, H, H) -> (32, H, H)
#         self.reduce_y = nn.Conv2d(B, 32, kernel_size=1)

#         # Merge features: (32 + 3, H, H) -> (c, H, H)
#         self.merge = nn.Sequential(
#             nn.Conv2d(32 + 3, 32, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(32, c, kernel_size=1),
#         )

#     def forward(self, features, y):
#         """
#         Args:
#             features: (1, 768)
#             y: (B, H, H)
#         Returns:
#             output: (c, H, H)
#         """
#         # Reshape features to (3, 16, 16)
#         features = features.view(3, 16, 16).unsqueeze(0)
#         upsampled = self.upsample1(features) # -> (3, 64, 64)
#         upsampled = self.upsample2(upsampled) # -> (3, H, H)

#         # If H is not divisible by 64, crop or interpolate
#         if upsampled.shape[2] != self.H:
#             upsampled = F.interpolate(upsampled, size=(self.H, self.H), mode='bilinear')

#         y = y.unsqueeze(0)
#         reduced_y = self.reduce_y(y) # (32, H, H)

#         concatenated = torch.cat([reduced_y, upsampled], dim=1)
#         output = self.merge(concatenated) # (c, H, H)
#         return output.squeeze(0)
