import torch.nn as nn
import torch
import sys
import argparse

global_path = "/home/ids/edabier/HSU"
sys.path.append(global_path)

from spectral_earth.src.backbones.spec_vit import SpecViTBase

sys.path.append(f"{global_path}/IEEE_TPAMI_SpectralGPT")
import models_mae_spectral

sys.path.append(f"{global_path}/HyperFree")
from HyperFree import build_HyperFree_vit_b, predictor

sys.path.append(f"{global_path}/DOFA")
from dofa_v1 import vit_base_patch16

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    print(f"{torch.cuda.is_available()}")
    dev = "cpu"


class FoundationModel():
    def __init__(self, model_name, patch_size=None, im_size=None, channels=None):
        """
        Instantiate the provided foundation model
        """
        models = ["SpectralEarth", "SpectralGPT", "DOFA", "HyperFree", "HyperSL", "HyperSIGMA"]

        if model_name not in models:
            raise ValueError("The provided model_name does not correspond to any of [SpectralEarth, SpectralGPT, DOFA, HyperFree, HyperSL, HyperSIGMA]")

        self.model_name = model_name
        self.patch_size = patch_size
        self.im_size = im_size

        if model_name == "SpectralEarth":
            model = SpecViTBase()
            model.vit_core.head = torch.nn.Identity()
            state_dict = torch.load(f"{global_path}/spectral_earth/data/data/spec_ViTb_mae.pth")
            model.load_state_dict(state_dict, strict=False)

            self.model = model
            self.im_size = 128

        elif model_name == "SpectralGPT":
            assert channels is not None, "The number of channels must be specified for SpectralGPT"
            state_dict = torch.load(f"{global_path}/IEEE_TPAMI_SpectralGPT/data/SpectralGPT+.pth", map_location=dev, weights_only=False)["model"]
            self.im_size = 128
            spectral_gpt = models_mae_spectral.mae_vit_base_patch8_128(num_frames=channels, pred_t_dim=channels)
            # state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            # spectral_gpt.load_state_dict(state_dict, strict=False)
            encoder_keys = [k for k in state_dict.keys() if k.startswith('patch_embed') or k.startswith('blocks') or k.startswith('norm')]
            encoder_state_dict = {k: state_dict[k] for k in encoder_keys}
            spectral_gpt.load_state_dict(encoder_state_dict, strict=False)
                        
            self.model = spectral_gpt
            
        elif model_name == "DOFA":
            state_dict = torch.load("/home/ids/edabier/HSU/DOFA/checkpoints/DOFA_ViT_base_e100.pth", map_location=dev)
            self.im_size = 224
            model = vit_base_patch16()
            model.load_state_dict(state_dict, strict=False)
            self.model = model

        elif model_name == "HyperFree":
            assert self.patch_size is not None, "Patch_size must be set"
            assert self.im_size is not None, "im_size must be set"

            pred = build_HyperFree_vit_b(checkpoint="/home/ids/edabier/HSU/HyperFree/data/HyperFree-b.pth", image_size=im_size, vit_patch_size=patch_size)
            self.model = predictor.HyperFree_Predictor(pred)

        elif model_name == "HyperSL":
            pass

        elif model_name == "HyperSigma":
            pass
        
    def create_decoder(self, c, B):
        self.decoder = Decoder(c, B)

    def get_features(self, Y, wavelengths=None):
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
        
        if H > self.im_size:
            print(f"Input HSI larger than expected size ({self.im_size}), cutting it to match")
            Y = Y[:, :, :self.im_size, :self.im_size]
        elif H < self.im_size:
            # TO DO
            print(f"Input HSI smaller than expected size ({self.im_size}), padding to match")

        if self.model_name == "HyperFree":
            assert wavelengths is not None, "HyperFree needs wavelengths list for spectral embedding"
            GSD = 0.456
            ratio = 1024 / (max(Y.shape[2], Y.shape[3]))
            GSD = GSD / ratio
            GSD = torch.tensor([GSD])

            input_im = self.model.transform.apply_image_torch(Y)
            self.model.set_torch_image(input_im, original_image_size=(Y.shape[1], Y.shape[2]), spectral_lengths=wavelengths, GSD=GSD)

            return self.model.features
        
        elif self.model_name == "DOFA":
            assert wavelengths is not None, "DOFA needs wavelengths list for [?]"
            return self.model.forward_features(Y, wave_lists=wavelengths) 
        
        elif self.model_name == "SpectralEarth":
            return self.model(Y)
        
        else:
            pass
    
    def get_abundances(self, F):
        # TO DO: define a method to eYtract A from features F
        pass

    def unmix(self, Y):
        F = self.get_features(Y)
        A = self.get_abundances(F)
        Y_hat = self.decoder(A)
        E = self.decoder.get_endmembers()

        return E, A, Y_hat

class weightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.clamp_(min=0)

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
        constraints = weightConstraint()
        self.decoder.apply(constraints)
        return self.decoder.weight.data