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
from src.models import models

import logging

def train_model(Y, fm, B, c, D, alpha, new_H, wavelengths, dev):
    epochs, lr = 200, 0.002
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.18),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
    
    model = rsfm.Unmixing_from_features(D=D, alpha=alpha, H=new_H, B=B, c=c, n_features=1, use_cls=False, is_cnnaeu=False)
    compiled_model = torch.compile(model)
    compiled_model.apply(model.weights_init)
    compiled_model = models.init_decoder_weights(compiled_model, Y/Y.max(), c, is_unmixer=True, normalize=False)

    compiled_model = compiled_model.to(dev)
    optimizer = build_optim_wrapper(compiled_model, optim_wrapper)

    for epoch in range(epochs):

        optimizer.zero_grad()
        Y = Y.to(dev)
        
        _, features = rsfm.extract_f(fm, Y, new_H, wavelengths, use_cls=False)
        E_hat, A_hat, Y_hat = compiled_model(features)
        loss = compiled_model.loss(Y, Y_hat, A_hat, E_hat)
        
        loss.backward()
        nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=10, norm_type=1)
        optimizer.step()

        with torch.no_grad():
            constraints = models.weightConstraint()
            compiled_model.decoder.apply(constraints)

    compiled_model.eval()

    with torch.no_grad():
        _, features = rsfm.extract_f(fm, Y, new_H, wavelengths, use_cls=False)
        E_hat, A_hat, _ = compiled_model(features)
    
    return E_hat, A_hat

def run_one_xp(i_dataset, i_pad, pad, i_xp, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, B, c, H, dev):
    
    fm, Y_up, _ = rsfm.create_fm("DOFA", Y_init, n_features=1, use_cls=False, extend_cls=False, path=global_path, patch_size=16)
    features = rsfm.get_dofa_features(fm, Y_up, wavelengths)
    D = int(features.shape[0]/1)
    alpha = int(features.shape[1]**0.5)

    Y_padded = F.pad(Y_init, pad=(pad, pad, pad, pad), mode="reflect")
    A_hat_padded = torch.zeros(1, c, H+2*pad,H+2*pad)
    weights = torch.zeros_like(A_hat_padded)

    E_hats = torch.zeros((2*pad+1)**2, B, c)

    for i in range(2*pad + 1):
        for j in range(2*pad + 1):

            Y_crop = Y_padded[:, :, i:i+H, j:j+H]
            E_hat, A_hat = train_model(Y_crop, fm, B, c, D, alpha, H, wavelengths, dev)

            E_hats[i+j] = E_hat
            A_hat_padded[:, :, i:i+H, j:j+H] += A_hat
            weights[:, :, i:i+H, j:j+H] += 1

    A_hat_padded /= weights
    E_hat_m = torch.mean(E_hats, dim=0)

    sad_pad, _, mse_pad = plots.compute_metrics_and_plot(E_hat_m, A_hat_padded[:,:,pad*2:-1-2*pad, pad*2:-1-2*pad], A_init[:,pad:-1-pad, pad:-1-pad], E_init, normalize_E=True, normalize_A=True, return_results=True, plot_A=False, plot_E=False)
    mse_tensor[i_dataset, i_pad, i_xp] = mse_pad
    sad_tensor[i_dataset, i_pad, i_xp] = sad_pad

    return mse_tensor, sad_tensor

def main(dev):

    datasets = ["urban"] #["samson", "apex", "jasper", "urban"]
    paddings = range(6)
    n_xp = 10

    mse_tensor = torch.zeros(len(datasets), len(paddings), n_xp)
    sad_tensor = torch.zeros(len(datasets), len(paddings), n_xp)

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

        mean_mse, std_mse, mean_sad, std_sad = torch.zeros(len(paddings)), torch.zeros(len(paddings)), torch.zeros(len(paddings)), torch.zeros(len(paddings))

        for i_pad, pad in enumerate(paddings):

            print(f"Training DOFA with padding = {pad}")

            for i_xp in range(n_xp):

                print(f"------ Running {i_xp+1}th experiment ------")

                mse_tensor, sad_tensor = run_one_xp(i_dataset, i_pad, pad, i_xp, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, B, c, H, dev)
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