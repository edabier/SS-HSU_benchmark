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
from src.models import unmixers as unmx
from src.models import models

import logging

def instantiate_model(Y, wavelengths, version="v2", size="large"):

    fm, Y_init_fm, new_H = rsfm.create_fm("DOFA", Y, size=size, version=version, path=global_path)
    features_dofa = rsfm.get_dofa_features(fm, Y_init_fm, wavelengths)
    D = int(features_dofa.shape[0])
    alpha = int(features_dofa.shape[1]**0.5)

    return fm, Y_init_fm, new_H, D, alpha

def upsample_features(fm, Y, wavelengths, new_H, D, alpha):
    patch = new_H//alpha
    padding = patch//2
    x_new = torch.linspace(0, new_H - 1, new_H)
    grid_y, grid_x = torch.meshgrid(x_new, x_new, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    grid = grid / torch.tensor([(new_H - 1) / 2, (new_H - 1) / 2]) - 1

    Y_padded = F.pad(Y, pad=(padding, padding, padding, padding), mode="reflect")
    feature_map = torch.zeros(1, D, new_H, new_H, device=dev)

    with torch.no_grad():
        for i in range(0, 2*padding):
            for j in range(0, 2*padding):
                
                Y_crop = Y_padded[:, :, i:i+new_H, j:j+new_H]
                Y_crop = Y_crop.to(dev)
                _, features = rsfm.extract_f(fm, Y_crop, new_H, wavelengths)

                features = utils.oneD_to_2d(features)
                feature_map[:,:, i::patch, j::patch] = features
                
                del features, Y_crop
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
    return feature_map

def run_one_xp(i_dataset, i_xp, n_train, mse_tensor, sad_tensor, Y_init_fm, A_init_fm, E_init, features_map, D, new_H, dev):

    Y_init_fm = Y_init_fm.to(dev)
    A_init_fm = A_init_fm.to(dev)
    E_init = E_init.to(dev)
    B, c = E_init.shape

    epochs, lr = 200, 0.002
    optim_wrapper = dict(
        optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.18),
        constructor='LayerDecayOptimizerConstructor_ViT',
        paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9, ))

    E_hats, A_hats = torch.zeros(n_train, B, c), torch.zeros(n_train, c, new_H, new_H)

    for i in range(n_train): 
        print(f"training {i}/{n_train}")

        model = unmx.UnmixingFromUpFeat(D=D, H=new_H, B=B, c=c)
        model.apply(model.weights_init)
        model = models.init_decoder_weights(model, Y_init_fm/Y_init_fm.max(), c, is_unmixer=True, normalize=False)
        model = model.to(dev)
        optimizer = build_optim_wrapper(model, optim_wrapper)

        for epoch in range(epochs):
            optimizer.zero_grad()
            E_hat, A_hat, Y_hat = model(features_map)
            E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)
            A_hat = A_hat.unsqueeze(0)
            loss = model.loss(Y_init_fm, Y_hat, A_hat, E_hat)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                constraints = models.weightConstraint()
                model.decoder.apply(constraints)
            
            del E_hat, A_hat
                    
        model.eval()
        with torch.no_grad():    
            E_hat, A_hat, Y_hat = model(features_map)
            E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)

            E_hats[i] = E_hat
            A_hats[i] = A_hat.squeeze(0)
            del E_hat, A_hat
            
    E_hat_m = torch.mean(E_hats, dim=0)
    A_hat_m = torch.mean(A_hats, dim=0)
    sad, _, mse = plots.compute_metrics_and_plot(E_hat_m, A_hat_m, A_init_fm, E_init, normalize_E=True, normalize_A=True, return_results=True, plot_E=False, plot_A=False)

    mse_tensor[i_dataset, i_xp] = mse
    sad_tensor[i_dataset, i_xp] = sad

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    return mse_tensor, sad_tensor

def main(args, dev):
    n_xp = args.n_xp
    n_train = args.n_train
    size = args.size
    version = args.version

    print(f"Running {n_train} xp of {n_train} trainings of DOFA {version}-{size}")

    datasets = ["apex", "jasper", "urban", "samson", "urban4"]
    # datasets = ["urban4"]

    # shape (n_datasets, step, n_xp)
    mse_tensor = torch.zeros(len(datasets), n_xp)
    sad_tensor = torch.zeros(len(datasets), n_xp)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        A_init = torch.tensor(data["A"], dtype=torch.float)

        Y_init = utils.oneD_to_2d(Y_flat).unsqueeze(0)
        A_init = utils.oneD_to_2d(A_init).unsqueeze(0)
        H = Y_init.shape[-1]

        if dataset == "urban4":
            wavelengths_path = f"{global_path}/SS-HSU_benchmark/datasets/urban_wavelength.txt"
        else:
            wavelengths_path = f"{global_path}/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt"
        with open(wavelengths_path, "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]
                
        fm, Y_init_fm, new_H, D, alpha = instantiate_model(Y_init, wavelengths, version, size)
        _, A_init_fm = rsfm.reshape_Y("DOFA", Y_init, new_H, A_init) 
        features_map = upsample_features(fm, Y_init_fm, wavelengths, new_H, D, alpha)
        features_map = (features_map - features_map.mean())/ (features_map.std() + 1e-8)

        for i_xp in range(n_xp):

            print(f"------ Running {i_xp+1}th experiment ------")
            mse_tensor, sad_tensor = run_one_xp(i_dataset, i_xp, n_train, mse_tensor, sad_tensor, Y_init_fm, A_init_fm, E_init, features_map, D, new_H, dev)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
        mse = mse_tensor[i_dataset]
        sad = sad_tensor[i_dataset]
        
        wandb.log({f"{dataset}_DOFA_MSE_mean": torch.mean(mse)})
        wandb.log({f"{dataset}_DOFA_MSE_std": torch.std(mse)})
        wandb.log({f"{dataset}_DOFA_SAD_mean": torch.mean(sad)})
        wandb.log({f"{dataset}_DOFA_SAD_std": torch.std(sad)})

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.WARNING) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_xp", default=10, type=int)
    parser.add_argument("--n_train", default=15, type=int)
    parser.add_argument("--size", default="large", type=str)
    parser.add_argument("--version", default="v2", type=str)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        
    run = wandb.init(project=f"DOFA_upsampled_training")
    
    print(f"Starting project on dev: {dev}")
    
    main(args, dev)