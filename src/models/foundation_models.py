import torch.nn as nn
import torch.nn.functional as F
import torch
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

sys.path.append(f"{global_path}/DOFA")
from dofa_v1 import vit_base_patch16

sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from models.model import HyperSIGMA_Unmix

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    print(f"{torch.cuda.is_available()}")
    dev = "cpu"


MODELS = ["SpectralEarth", "SpectralGPT", "DOFA", "HyperFree", "HyperSL", "HyperSIGMA"]

class FoundationModel(nn.Module):
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

def features_comparison(Y_gt, Y_hat):
    """
    TO DO: write a function that computes the similarity between the features
    Extracted from Y_gt and those extraced from Y_hat by some RSFM
    Use Feed or LIPS metrics?
    """
    
    
    pass

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
    
class Unmixing_from_features(nn.Module):
    def __init__(self, D, B, c):
        super(Unmixing_from_features, self).__init__()
        self.D = D
        self.B = B
        self.c = c

        self.upsample = nn.ConvTranspose2d( # upsamples 14 -> 224
            in_channels=4*D,
            out_channels=4*D,
            kernel_size=3,
            stride=17,
            padding=0,
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(D*4, D*2, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(D*2),
            nn.Dropout(0.2),
            nn.Conv2d(D*2,  D, kernel_size=(1, 1)) )
        self.conv2 = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=1),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2),
        )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_ab=0.35, W_tv=0.1):
        sad = utils.SADLoss()
        loss_sad = sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()
        loss_tv = W_tv * (torch.abs(E_hat[:, 1:] - E_hat[:, :(-1)]).sum())
        loss = 100*loss_sad + loss_ab + loss_tv
        return loss
    
    def get_abundances(self, features):

        H_p = W_p = int(features.shape[1] ** 0.5)
        features_2d = features.view(
            1, 4*self.D, H_p, W_p
        )
        features_up = self.upsample(features_2d)
        patch_D = self.conv1(features_up)
        A_hat   = self.conv2(patch_D)
        A_hat   = self.sum_to_one(A_hat)

        return A_hat
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        # Y_hat = Y_hat.reshape(1, self.B, 224**2)
        # A_hat = A_hat.reshape(1, self.c, 224**2)

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

#         H_p = W_p = int(features.shape[1] ** 0.5)
#         features_2d = features.view(
#             1, 4*self.D, H_p, W_p
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
