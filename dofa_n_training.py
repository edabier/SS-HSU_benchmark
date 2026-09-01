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

def instantiate_model(Y, wavelengths, version="v1", size="large"):
    fm, Y_init_fm, new_H = rsfm.create_fm("DOFA", Y, size=size, version=version, path=global_path)
    features_dofa = rsfm.get_dofa_features(fm, Y_init_fm, wavelengths)
    D = int(features_dofa.shape[0])
    alpha = int(features_dofa.shape[1]**0.5)

    return fm, Y_init_fm, new_H, D, alpha

def run_one_xp(i_dataset, i_train, n_train, i_xp, dataset, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, H, dev):

    Y_init = Y_init.to(dev)
    A_init = A_init.to(dev)
    E_init = E_init.to(dev)
    B, c = E_init.shape

    fm, Y_init_fm, new_H, D, alpha = instantiate_model(Y_init, wavelengths, H)
    sads, mses = [], []
    E_hats, A_hats = torch.zeros(n_train, B, c), torch.zeros(n_train, c, new_H, new_H)

    for i in range(n_train): 
        print(f"training {i}/{n_train}")

        model = unmx.UnmixingFromFeatures2(D=D, alpha=alpha-2, B=B, c=c)
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

                Y_fm, features = rsfm.extract_f(fm, Y, new_H, wavelengths)
                E_hat, A_hat, Y_hat, _ = model(features)
                
                loss = model.loss(Y_fm, Y_hat, A_hat, E_hat)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)
                    
        model.eval()
        
        with torch.no_grad():
            _, A_init_fm, features = rsfm.extract_f(fm, Y_init, new_H, wavelengths, A_init)
            E_hat1, A_hat1, Y_hat1, _ = model(features)

        E_hats[i] = E_hat1
        A_hats[i] = A_hat1.squeeze(0)

    E_hat_m = torch.mean(E_hats, dim=0)
    A_hat_m = torch.mean(A_hats, dim=0)

    assert not E_hat_m.isnan().any(), "E_hat_m has nan values"

    sad, _, mse = plots.compute_metrics_and_plot(E_hat_m, A_hat_m, A_init_fm, E_init, normalise_E=True, normalise_A=True, return_results=True, plot_E=False, plot_A=False)
    print(f"Average SAD = {format(sad, '.3f')}, MSE = {format(mse, '.3f')}")

    mse_tensor[i_dataset, i_train, i_xp] = mse
    sad_tensor[i_dataset, i_train, i_xp] = sad

    return mse_tensor, sad_tensor

def main(args, dev):
    n_xp = 10 #args.n_xp
    step = 5 #args.step

    datasets = ["samson", "urban"]

    # shape (n_datasets, step, n_xp)
    trainings = [1]
    trainings += [i for i in range(5,25,step)]
    mse_tensor = torch.zeros(len(datasets), len(trainings), n_xp)
    sad_tensor = torch.zeros(len(datasets), len(trainings), n_xp)

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

        mean_sad = torch.zeros(len(trainings))
        std_sad = torch.zeros(len(trainings))
        mean_nmse = torch.zeros(len(trainings))
        std_nmse = torch.zeros(len(trainings))

        for idx_train, n_train in enumerate(trainings):

            print(f"Training DOFA {n_train} times")

            for i_xp in range(n_xp):

                print(f"------ Running {i_xp+1}th experiment ------")
                mse_tensor, sad_tensor = run_one_xp(i_dataset, idx_train, n_train, i_xp, dataset, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, H, dev)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
            
            # metrics_i = (mse_tensor[i_dataset, idx_train] + sad_tensor[i_dataset, idx_train])/2
            mean_sad[idx_train] = torch.mean(sad_tensor[i_dataset, idx_train], dim=0)
            std_sad[idx_train] = torch.std(sad_tensor[i_dataset, idx_train], dim=0)
            mean_nmse[idx_train] = torch.mean(mse_tensor[i_dataset, idx_train], dim=0)
            std_nmse[idx_train] = torch.std(mse_tensor[i_dataset, idx_train], dim=0)
            
            wandb.log({f"{dataset}_SAD_{n_train}_mean": mean_sad[idx_train]})
            wandb.log({f"{dataset}_SAD_{n_train}_std": std_sad[idx_train]})
            wandb.log({f"{dataset}_NMSE_{n_train}_mean": mean_nmse[idx_train]})
            wandb.log({f"{dataset}_NMSE_{n_train}_std": std_nmse[idx_train]})

        # torch.save(mean_metrics, f"/home/ids/edabier/HSU/SS-HSU_benchmark/dofa_n_trains/{dataset}_DOFA_{idx_train}_mean.pt")
        # torch.save(std_metrics, f"/home/ids/edabier/HSU/SS-HSU_benchmark/dofa_n_trains/{dataset}_DOFA_{idx_train}_std.pt")

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.WARNING) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_xp", default=10, type=int)
    parser.add_argument("--step", default=5, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        
    run = wandb.init(project=f"DOFA_n_training")
    
    print(f"Starting project on dev: {dev}")
    
    main(args, dev)