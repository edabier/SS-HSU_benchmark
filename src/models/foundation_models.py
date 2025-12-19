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

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    print(f"{torch.cuda.is_available()}")
    dev = "cpu"


class FoundationModel():
    def __init__(self, model_name, patch_size=None, im_size=None):
        """
        Instantiate the provided foundation model
        """
        self.model_name = model_name

        if model_name == "SpectralEarth":
            model = SpecViTBase()
            model.vit_core.head = torch.nn.Identity()
            state_dict = torch.load(f"{global_path}/spectral_earth/data/data/spec_ViTb_mae.pth")
            model.load_state_dict(state_dict, strict=False)

            self.model = model
            self.im_size = 128

        elif model_name == "HyperFree":
            assert patch_size is None, "Patch_size must be set"
            assert im_size is None, "im_size must be set"

            pred = build_HyperFree_vit_b(checkpoint="/home/ids/edabier/HSU/HyperFree/data/HyperFree-b.pth", image_size=im_size, vit_patch_size=patch_size)
            self.model = predictor.HyperFree_Predictor(pred)

        elif model_name == "HyperSL":
            pass

        elif model_name == "HyperSigma":
            pass

        elif model_name == "SpectralGPT":
            checkpoint = torch.load(f"{global_path}/IEEE_TPAMI_SpectralGPT/data/SpectralGPT+.pth", map_location=dev, weights_only=False)["model"]
            img_size = 128
            patch_size = 8
            spectral_gpt = models_mae_spectral.__dict__[f"mae_vit_base_patch{patch_size}_{img_size}"]()
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            spectral_gpt.load_state_dict(state_dict, strict=False)
            
            self.model = spectral_gpt
            self.im_size = img_size
        
    def create_decoder(self, c, B):
        self.decoder = Decoder(c, B)

    def get_features(self, x, wavelengths=None):
        """
        Forwards the input HSI to the model's encoder to get features

        Args:
            x: input HSI tensor of shape (batch, B, H, W)
            for HyperFree, H and W must be multiples of batch_size=8 and values must be normalized (/max(Y))
        """
        if x.dim() == 4:
            batch, B, H, W = x.shape
        elif x.dim() == 3:
            B, H, W = x.shape
            x = x.unsqueeze(0)
        else:
            print("Input HSI must be of shape (B, H, W) or (batch, B, H, W)")
        
        if H > self.im_size:
            print(f"Input HSI larger than expected size ({self.im_size}), cutting it to match")
            x = x[:, :, :self.im_size, :self.im_size]
        elif H < self.im_size:
            # TO DO
            print(f"Input HSI smaller than expected size ({self.im_size}), padding to match")

        if self.model_name == "HyperFree":
            assert wavelengths is None, "HyperFree needs wavelengths list for spectral embedding"
            GSD = 0.456
            ratio = 1024 / (max(x.shape[2], x.shape[3]))
            GSD = GSD / ratio
            GSD = torch.tensor([GSD])

            input_im = self.model.transform.apply_image_torch(x)
            self.model.set_torch_image(input_im, original_image_size=(x.shape[1], x.shape[2]), spectral_lengths=wavelengths, GSD=GSD)
        else:
            return self.model(x)
    
    def get_abundances(self, F):
        # TO DO: define a method to extract A from features F
        pass

    def unmix(self, x):
        F = self.get_features(x)
        A = self.get_abundances(F)
        x_hat = self.decoder(A)
        E = self.decoder.get_endmembers()

        return E, A, x_hat

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