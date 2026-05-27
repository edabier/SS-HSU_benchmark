import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
import matplotlib.pyplot as plt
import os
import sys
import gc
import wandb

from mmengine.optim import build_optim_wrapper

global_path = "/home/ids/edabier/HSU"
sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from mmcv__custom import layer_decay_optimizer_constructor_vit

from src.utils import utils
from src.utils import plots
from src.models import foundation_models as rsfm
from src.models import unmixers as unmx
from src.models import models

import logging

def get_adjacent_crops(A_hats, shift):
    """
    A_hats: list of estimated abundances, each of shape (C, H, W)
    pad: padding size
    shift: shift between crops

    Returns: list of tuples (overlap_k, overlap_l) for adjacent crops
    """

    overlaps = []
    n_crops = len(A_hats)
    n_rows = n_cols = int(n_crops**0.5)
    H = W = A_hats[0].shape[-1]

    # Directions: (di, dj) for adjacent crops (right, bottom, bottom-right, bottom-left)
    # These are the 4 possible directions for adjacency in a 2D grid
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for i in range(n_rows):
        for j in range(n_cols):
            k = i * n_rows + j  # index of current crop
            i_k = i * shift
            j_k = j * shift
            # Check all 4 possible adjacent crops
            for di, dj in directions:
                i_adj = i + di
                j_adj = j + dj
                if 0 <= i_adj < n_rows and 0 <= j_adj < n_cols:
                    l = i_adj * n_rows + j_adj  # index of adjacent crop
                    i_l = i_adj * shift
                    j_l = j_adj * shift
                    # Compute overlap
                    i_start = max(i_k, i_l)
                    i_end = min(i_k + H, i_l + H)
                    j_start = max(j_k, j_l)
                    j_end = min(j_k + W, j_l + W)
                    if i_end > i_start and j_end > j_start:
                        # Extract overlapping regions
                        overlap_k = A_hats[k][:, i_start - i_k:i_end - i_k, j_start - j_k:j_end - j_k]
                        overlap_l = A_hats[l][:, i_start - i_l:i_end - i_l, j_start - j_l:j_end - j_l]
                        overlaps.append((overlap_k, overlap_l))
    return overlaps

def consistency_loss(overlaps, similarity="mse"):
    """
    overlaps: list of tuples (overlap_k, overlap_l)
    similarity: "mse" or "cosine"
    Returns: total consistency loss
    """

    total_loss = 0.0

    for overlap_k, overlap_l in overlaps:

        if similarity == "mse":
            loss = F.mse_loss(overlap_k, overlap_l)

        elif similarity == "cosine":
            loss = 1 - F.cosine_similarity(overlap_k, overlap_l).mean()

        else:
            raise ValueError("Similarity must be 'mse' or 'cosine'")
        total_loss += loss

    return total_loss

def instantiate_models(pad, shift, Y_init, D, alpha, new_H, B, c, dev):

    n_models = (2*(pad-(shift-1))+1)**2
    n_rows = 2*(pad-(shift-1))+1

    model_list = []
    optimizers = []

    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=0.002, betas=(0.9, 0.999), weight_decay=0.18),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

    for i in range(n_models):
        model = torch.compile(unmx.UnmixingFromFeatures(D=D, alpha=alpha, H=new_H, B=B, c=c))
        model.apply(model.weights_init)

        x_i = (i//n_rows)
        y_i = (i%n_rows)*shift
        Y = Y_init[:, :, x_i:x_i+new_H, y_i:y_i+new_H]

        model = models.init_decoder_weights(model, Y/Y.max(), c, is_unmixer=True, normalize=False)
        model = model.to(dev)
        optimizer = build_optim_wrapper(model, optim_wrapper)

        model_list.append(model)
        optimizers.append(optimizer)
    
    return model_list, optimizers

def run_one_xp(i_dataset, i_xp, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, B, c, H, dev):
    pad = 8
    shift = 8
    n_rows = 2*(pad-(shift-1))+1

    fm, Y_up, _ = rsfm.create_fm("DOFA", Y_init, n_features=1, use_cls=False, extend_cls=False, path=global_path, patch_size=16)
    features = rsfm.get_dofa_features(fm, Y_up, wavelengths)
    D = int(features.shape[0]/1)
    alpha = int(features.shape[1]**0.5)

    Y_padded = F.pad(Y_init, pad=(pad, pad, pad, pad), mode="reflect")

    ################ Training ################

    epochs, lr = 200
    model_list, optimizers = instantiate_models(pad, shift, Y_padded, D, alpha, H, B, c, dev)
    
    for epoch in range(epochs):

        total_loss = 0
        A_hats = []


        for i in range(0, 2*pad + 1, shift):
            for j in range(0, 2*pad + 1, shift):

                i_model = n_rows*(i//shift) + j//shift
                model = model_list[i_model]
                optimizer = optimizers[i_model]
                optimizer.zero_grad()

                Y_crop = Y_padded[:, :, i:i+H, j:j+H]
                Y_crop = Y_crop.to(dev)

                _, features = rsfm.extract_f(fm, Y_crop, H, wavelengths, use_cls=False)
                E_hat, A_hat, Y_hat = model(features)
                A_hats.append(A_hat)
                
                loss = model.loss(Y_crop, Y_hat, A_hat, E_hat)
                total_loss += loss
        
        overlaps = get_adjacent_crops(A_hats, shift)
        consistency = consistency_loss(overlaps)

        total_loss += consistency
        total_loss.backward()

        for i_model, model in enumerate(model_list):
            optimizer = optimizers[i_model]    
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
            optimizer.step()

            with torch.no_grad():
                constraints = models.weightConstraint()
                model.decoder.apply(constraints)

    ################ Evaluation ################

    A_hat_padded = torch.zeros(1, c, H+2*pad,H+2*pad)
    weights = torch.zeros_like(A_hat_padded)
    E_hats = torch.zeros((2*pad+1)**2, B, c)

    for i in range(0, 2*pad + 1, shift):
        for j in range(0, 2*pad + 1, shift):

            i_model = n_rows*(i//shift) + j//shift
            model = model_list[i_model]
            optimizer = optimizers[i_model]
            optimizer.zero_grad()

            model.eval()
            with torch.no_grad():

                Y_crop = Y_padded[:, :, i:i+H, j:j+H]
                _, features = rsfm.extract_f(fm, Y_crop, H, wavelengths, use_cls=False)
                E_hat, A_hat, _ = model(features)
    
                E_hats[i+j] = E_hat
                A_hat_padded[:, :, i:i+H, j:j+H] += A_hat
                weights[:, :, i:i+H, j:j+H] += 1

    A_hat_padded /= weights
    E_hat_m = torch.mean(E_hats, dim=0)

    sad_pad, _, mse_pad = plots.compute_metrics_and_plot(E_hat_m, A_hat_padded[:,:,pad*2:-1-2*pad, pad*2:-1-2*pad], A_init[:,pad:-1-pad, pad:-1-pad], E_init, normalize_E=True, normalize_A=True, return_results=True, plot_A=False, plot_E=False)
    mse_tensor[i_dataset, i_xp] = mse_pad
    sad_tensor[i_dataset, i_xp] = sad_pad

    return mse_tensor, sad_tensor

def main(dev):

    datasets = ["samson", "apex", "jasper", "urban"]
    n_xp = 10

    mse_tensor = torch.zeros(len(datasets), n_xp)
    sad_tensor = torch.zeros(len(datasets), n_xp)

    for i_dataset, dataset in enumerate(datasets):
        data = io.loadmat(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float)
        A_flat = torch.tensor(data["A"], dtype=torch.float)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        B, c = E_init.shape[0], E_init.shape[1]

        Y_init = utils.oneD_to_2d(Y_flat)
        A_init = utils.oneD_to_2d(A_flat)
        H = Y_init.shape[-1]
        Y_init = Y_init.unsqueeze(0)

        with open(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt", "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]


            print(f"Training DOFA")

            for i_xp in range(n_xp):

                print(f"------ Running {i_xp+1}th experiment ------")

                mse_tensor, sad_tensor = run_one_xp(i_dataset, i_xp, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, B, c, H, dev)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            
            mean_mse[i_pad] = torch.mean(mse_tensor[i_dataset, i_pad])
            std_mse[i_pad] = torch.std(mse_tensor[i_dataset, i_pad])

            mean_sad[i_pad] = torch.mean(sad_tensor[i_dataset, i_pad])
            std_sad[i_pad] = torch.std(sad_tensor[i_dataset, i_pad])
            
            wandb.log({f"{dataset}_DOFA_{i_pad}_mean_mse": mean_mse[i_pad]})
            wandb.log({f"{dataset}_DOFA_{i_pad}_std_mse": std_mse[i_pad]})

            wandb.log({f"{dataset}_DOFA_{i_pad}_mean_sad": mean_sad[i_pad]})
            wandb.log({f"{dataset}_DOFA_{i_pad}_std_sad": std_sad[i_pad]})

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.WARNING) 

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        
    run = wandb.init(project=f"DOFA_n_padding")
    
    print(f"Starting project on dev: {dev}")
    
    main(dev)