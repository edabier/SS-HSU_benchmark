import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as io
import sys
import gc
import os
import wandb

global_path = "/home/ids/edabier/HSU"
# global_path = "/home/edabier/Documents/Thèse/benchmark"
sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from mmengine.optim import build_optim_wrapper
from mmcv__custom import custom_layer_decay_optimizer_constructor

from src.utils import utils
from src.utils import plots
from src.models import models
from src.models import foundation_models as rsfm

def main(dev):
    datasets = ["samson", "jasper", "apex", "urban"]

    for dataset in datasets:

        print(f"Training {dataset}")
        data = io.loadmat(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float)
        A_flat = torch.tensor(data["A"], dtype=torch.float)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        B, c = E_init.shape

        Y_init = utils.oneD_to_2d(Y_flat)
        A_init = utils.oneD_to_2d(A_flat)
        Y_init = Y_init.unsqueeze(0)
        A = A_init.unsqueeze(0)

        wavelengths_path = f"{global_path}/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt"
        with open(wavelengths_path, "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]

        fm, Y_up, new_H = rsfm.create_fm("DOFA", Y_init, n_features=1, use_cls=False, extend_cls=False, path=global_path, patch_size=16)
        features = rsfm.get_dofa_features(fm, Y_up, wavelengths)
        D = int(features.shape[0]/1)
        alpha = int(features.shape[1]**0.5)

        Y_224, A_224 = rsfm.reshape_Y("DOFA", Y_init, new_H, A)
        x_new = torch.linspace(0, new_H - 1, new_H)
        grid_y, grid_x = torch.meshgrid(x_new, x_new, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        grid = grid / torch.tensor([(new_H - 1) / 2, (new_H - 1) / 2]) - 1

        patch = 16
        padding = patch//2
        epochs, lr = 200, 0.002
        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.18),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
            
        model = rsfm.Unmixing_from_features2(D=D, alpha=alpha, H=new_H, B=B, c=c, n_features=1)
        model.apply(model.weights_init)
        model = models.init_decoder_weights(model, Y_224/Y_224.max(), c, is_unmixer=True, normalize=False)
        model = model.to(dev)
        optimizer = build_optim_wrapper(model, optim_wrapper)

        Y_padded = F.pad(Y_224, pad=(padding, padding, padding, padding), mode="reflect")
        feature_map = torch.zeros(1, D, new_H, new_H, device=dev)
        with torch.no_grad():
            for i in range(0, 2*padding):
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

        for epoch in range(epochs):
            optimizer.zero_grad()
            feature_map_resampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            E_hat, A_hat, Y_hat = model(feature_map_resampled)
            E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)
            A_hat = A_hat.unsqueeze(0)
            loss = model.loss(Y_224, Y_hat, A_hat, E_hat)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                constraints = models.weightConstraint()
                model.decoder.apply(constraints)
            
            del E_hat, A_hat, feature_map, feature_map_resampled

            if epoch%(epochs//10)==0:
                print(f"Current epoch: {epoch}")
        
        model.eval()
        with torch.no_grad():
            feature_map_resampled = F.grid_sample(feature_map, grid, mode='bilinear', padding_mode='reflection', align_corners=True)
            E_hat, A_hat, Y_hat = model(feature_map_resampled)
            E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)
            A_hat = A_hat.unsqueeze(0)

        sad, _, mse = plots.compute_metrics_and_plot(E_hat, A_hat, A_224, E_init, normalize_E=True, normalize_A=True, return_results=True, save_mat=f"/home/ids/edabier/HSU/SS-HSU_benchmark/DOFA_results/{dataset}")
        wandb.log({f"{dataset}_DOFA_mse": mse})
        wandb.log({f"{dataset}_DOFA_sad": sad})
        
        del feature_map, E_hat, A_hat, E_init, A_224, Y_224, feature_map_resampled
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        
    run = wandb.init(project=f"DOFA_full_padding")
    
    print(f"Starting project on dev: {dev}")
    main(dev)