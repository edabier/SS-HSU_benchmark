import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
import argparse
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
from src.models import unmixers as unmx

import logging

def instantiate_model(model, Y, wavelengths, version="v1", size="large"):
    fm, Y_init_fm, new_H = rsfm.create_fm(model, Y, size=size, version=version, path=global_path)
    _, features = rsfm.extract_f(fm, Y_init_fm, new_H, wavelengths)
    D = int(features.shape[0])
    alpha = int(features.shape[1]**0.5)

    return fm, Y_init_fm, new_H, D, alpha

def run_one_xp(i_dataset, upsampler, model, i_xp, dataset, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, H, dev):

    Y_init = Y_init.to(dev)
    A_init = A_init.to(dev)
    E_init = E_init.to(dev)
    B, c = E_init.shape

    fm, Y_init_fm, new_H, D, alpha = instantiate_model(model, Y_init, wavelengths)
    n_train = 15
    E_hats, A_hats = torch.zeros(n_train, B, c), torch.zeros(n_train, c, new_H, new_H)

    for i in range(n_train): 
        
        print(f"Training {i+1}/{n_train}")

        model = unmx.UnmixingFromFeatures(D=D, alpha=alpha, B=B, c=c, H=new_H, upsampler=upsampler)

        model.apply(model.weights_init)
        model = models.init_decoder_weights(model, Y_init_fm/Y_init_fm.max(), c, is_unmixer=True)

        loader, _, _ = utils.create_dataloader(dataset, patch_size=H, dev=dev)
        epochs, lr = 200, 0.002

        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.18),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9, ))
        optimizer = build_optim_wrapper(model, optim_wrapper)

        for epoch in range(epochs):
            
            for Y, _, _ in loader:
        
                optimizer.zero_grad()

                Y = utils.oneD_to_2d(Y).to(dev)

                # Y_fm, features = rsfm.extract_f(fm, Y, new_H, wavelengths)
                Y_fm, features = rsfm.extract_f(fm, Y, new_H, wavelengths, patch_size=56)

                E_hat, A_hat, Y_hat = model(features, Y_fm)

                loss = model.loss(Y_fm, Y_hat, A_hat, E_hat)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)
                    
        model.eval()
        
        with torch.no_grad():
            _, A_init_fm, features = rsfm.extract_f(fm, Y, new_H, wavelengths, A_init)
            E_hat, A_hat, _ = model(features, Y_fm)

            if not E_hat.isnan().any().item() and not A_hat.isnan().any().item():
                sad, _, mse = plots.compute_metrics_and_plot(E_hat, A_hat, A_init_fm, E_init, normalise_E=True, normalise_A=True, return_results=True, plot_E=False, plot_A=False)
                print(f"Current SAD = {format(sad, '.3f')}, NMSE = {format(mse, '.3f')}")
            else:
                print(E_hat.isnan().any(), A_hat.isnan().any())

        E_hats[i] = E_hat
        A_hats[i] = A_hat.squeeze(0)

    valid_mask_E = ~torch.isnan(E_hats).any(dim=(1,2))
    valid_E_hats = E_hats[valid_mask_E]
    valid_mask_A = ~torch.isnan(A_hats).any(dim=(1,2,3))
    valid_A_hats = A_hats[valid_mask_A]

    for i in range(n_train):
        if E_hats[i].isnan().any():
            print(E_hats[i].isnan().any(), i)

    if valid_A_hats.shape[0] > 0:
        A_hat_m = torch.mean(valid_A_hats, dim=0)
    if valid_E_hats.shape[0] > 0:
        E_hat_m = torch.mean(valid_E_hats, dim=0)
        sad, _, mse = plots.compute_metrics_and_plot(E_hat_m, A_hat_m, A_init_fm, E_init, normalise_E=True, normalise_A=True, return_results=True, plot_E=False, plot_A=False)
        print(f"Average SAD = {format(sad, '.3f')}, MSE = {format(mse, '.3f')}")
    else:
        sad, mse = 0, 0
        print("No valid prediction for E_hat, all nans")

    mse_tensor[i_dataset, i_xp] = mse
    sad_tensor[i_dataset, i_xp] = sad

    return mse_tensor, sad_tensor

def main(args, dev):
    n_xp = 10
    upsampler = args.upsampler
    model = args.model

    # datasets = ["samson", "apex"]
    # datasets = ["jasper", "urban"]
    datasets = ["samson", "jasper", "apex", "urban"]

    # shape (n_datasets, n_xp)
    mse_tensor = torch.zeros(len(datasets), n_xp)
    sad_tensor = torch.zeros(len(datasets), n_xp)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float32)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        A_init = torch.tensor(data["A"], dtype=torch.float)

        Y_init = utils.oneD_to_2d(Y_flat).unsqueeze(0)
        A_init = utils.oneD_to_2d(A_init).unsqueeze(0)
        H = Y_init.shape[-1]

        with open(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt", "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]

        print(f"Training DOFA {upsampler} 15 times")

        for i_xp in range(n_xp):

            print(f"------ Running {i_xp+1}th experiment ------")
            mse_tensor, sad_tensor = run_one_xp(i_dataset, upsampler, model, i_xp, dataset, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, H, dev)
            
        mse = mse_tensor[i_dataset]
        sad = sad_tensor[i_dataset]
        
        wandb.log({f"{dataset}_{upsampler}_MSE_mean": torch.mean(mse)})
        wandb.log({f"{dataset}_{upsampler}_MSE_std": torch.std(mse)})
        wandb.log({f"{dataset}_{upsampler}_SAD_mean": torch.mean(sad)})
        wandb.log({f"{dataset}_{upsampler}_SAD_std": torch.std(sad)})

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.WARNING) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--upsampler", default="Features_fusion", type=str)
    parser.add_argument("--model", default="DOFA", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        
    run = wandb.init(project=f"{args.model}_{args.upsampler}")
    
    print(f"Starting project on dev: {dev}")
    
    main(args, dev)