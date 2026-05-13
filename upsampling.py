import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt
import scipy.io as io
import sys
from tqdm import tqdm
import gc

global_path = "/home/edabier/Documents/Thèse/benchmark"

from src.utils import utils
from src.models import foundation_models as rsfm

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
    print(f"Using device: {dev}")
  
else:
    dev = "cpu"
    print(f"Using device: {dev}")

""" Openning the image from which to extract features """
dataset = "urban"
data = io.loadmat(f"datasets/{dataset}.mat")
Y_flat = torch.tensor(data["Y"], dtype=torch.float)
A_flat = torch.tensor(data["A"], dtype=torch.float)
E_init = torch.tensor(data["E"], dtype=torch.float)
B, c, N = E_init.shape[0], E_init.shape[1], Y_flat.shape[1]

Y_init = utils.oneD_to_2d(Y_flat) # custom function to make a flat image 2D
A_init = utils.oneD_to_2d(A_flat)
H = Y_init.shape[-1]
Y_init = Y_init.unsqueeze(0)
A = A_init.unsqueeze(0)

wavelengths_path = f"{global_path}/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt"
with open(wavelengths_path, "r") as file:
    lines = file.readlines()
    wavelengths = [float(line.strip()) for line in lines if line.strip()]

fm_name = "DOFA"

# Here we just instantiate DOFA and reshape the input image to DOFA's expected input size (224, 224)
# DOFA needs the list of wavelengths of the image in input (specific to hyperspectral)
fm, Y_init_fm, new_H = rsfm.create_fm(fm_name, Y_init, size="base", version="v1", path=global_path)
features = rsfm.get_dofa_features(fm, Y_init_fm, wavelengths)

# My features are of shape (D=embed_dim, alpha²) where alpha is the patchified spatial dimension of the features
features = utils.oneD_to_2d(features) # Reshape the features to (D, alpha, alpha)
D = int(features.shape[0])
alpha = int(features.shape[1])

print(f"feature shape : {D, alpha, alpha}")

patch = new_H//alpha
padding = patch//2
shift = 1

Y_padded = F.pad(Y_init_fm, pad=(padding, padding, padding, padding), mode="reflect")
feature_map = torch.zeros(1, D, new_H, new_H, device=dev)

""" Actual upsampling loop """
with torch.no_grad():
    for i in tqdm(range(0, 2*padding)):
        for j in range(0, 2*padding):
            
            Y_crop = Y_padded[:, :, i:i+new_H, j:j+new_H]
            Y_crop = Y_crop.to(dev)
            _, features = rsfm.extract_f(fm, Y_crop, new_H, wavelengths, use_cls=False)

            features = features.reshape(1, D, alpha, alpha)
            feature_map[:,:, i::patch, j::patch] = features
            
            del features, Y_crop
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

""" To resample the upsampled features into an integer grid """
x_new = torch.linspace(0, new_H - 1, new_H)
grid_y, grid_x = torch.meshgrid(x_new, x_new, indexing='ij')
grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
grid = grid / torch.tensor([(new_H - 1) / 2, (new_H - 1) / 2]) - 1

feature_map_resampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='reflection', align_corners=True)