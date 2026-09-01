import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as io
import argparse
import matplotlib.pyplot as plt
import wandb
import os
import sys
import gc

from mmengine.optim import build_optim_wrapper

sys.path.append("/home/ids/edabier/HSU/HyperSIGMA/HyperspectralUnmixing")
from mmcv__custom import layer_decay_optimizer_constructor_vit

import src.utils.utils as utils
import src.models.foundation_models as rsfm
from src.models import unmixers as unmx
import src.training.training as training
import src.models.models as models

import logging

def instantiate_models(Y, wavelengths, B, c, H):

    model_list = {"DOFA": [], "SpecViT": [], "SpecAware": []}

    # DOFA
    if H < 224:
        Y_dofa = F.interpolate(Y, size=(224,224))
        H_dofa = 224
    else:
        Y_dofa = Y.clone()
        H_dofa = 224
    Y_dofa = Y_dofa[:,:,:H_dofa, :H_dofa]
    fm = rsfm.create_fm("DOFA", Y_dofa, version="v1", size="large")
    features_dofa = rsfm.get_dofa_features(fm, Y_dofa, wavelengths)
    D_dofa = int(features_dofa.shape[0])
    alpha = int(features_dofa.shape[1]**0.5)
    dofa = rsfm.Unmixing_from_features(D=D_dofa, alpha=alpha, B=B, c=c)
    model_list["DOFA"].append([fm, dofa])

    # SpecViT
    if H < 128:
        Y_specvit = F.interpolate(Y, size=(128,128))
        H_specvit = 128
    else:
        Y_specvit = Y.clone()
        H_specvit = 128
    Y_specvit = Y_specvit[:,:,:H_specvit, :H_specvit]
    fm = rsfm.create_fm("SpecViT", Y_specvit, size="base")
    features_specvit = rsfm.get_specvit_features(fm, Y_specvit)
    D_specvit = int(features_specvit.shape[0])
    alpha = int(features_specvit.shape[1]**0.5)
    specvit = rsfm.Unmixing_from_features(D=D_specvit, alpha=alpha, H=H_specvit, B=B, c=c)
    model_list["SpecViT"].append([fm, specvit])

    # SpecAware
    Y_specaware = Y[:, :, :H_dofa, :H_dofa]
    fm = rsfm.create_fm("SpecAware", Y_specaware)
    features_hyperfree = rsfm.get_hyperfree_features(fm, Y_specaware, wavelengths)
    D_hyperfree = features_hyperfree.shape[0]
    alpha = int(features_hyperfree.shape[1]**0.5)
    specaware = rsfm.Unmixing_from_features(D=D_hyperfree, alpha=alpha, B=B, c=c, H=H_dofa)
    model_list["SpecAware"].append([fm, specaware])


    return model_list, H_dofa

def transform_Y(Y, fm_name):
    batch, B, H, _ = Y.shape

    if fm_name == "DOFA":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:224, :224]
        return Y, new_H
            
    elif fm_name == "SpecAware":
        if H < 224:
            Y = F.interpolate(Y, size=(224,224))
            new_H = 224
        else:
            new_H = 224
        Y = Y[:,:,:224, :224]
        return Y, new_H

    elif fm_name == "SpecViT":
        if H < 128:
            Y = F.interpolate(Y, size=(128,128))
            new_H = 128
        else:
            new_H = 128
        Y = Y[:,:,:new_H, :new_H]
        return Y, new_H

def run_one_xp(n, dataset, mses, sads, mses_cnn, sads_cnn, Y_init, wavelengths, B, c, H, dev):

    model_names = ["DOFA", "SpecViT", "SpecAware"]
    models_list, H_hyperfree = instantiate_models(Y_init, wavelengths, B, c, H)

    for i_model, model_name in enumerate(model_names): 

        print(f"Training {model_name}")

        fm = models_list[f"{model_name}"][0][0]
        model = models_list[f"{model_name}"][0][1]
        model.apply(model.weights_init)
        Y, new_H = transform_Y(Y_init, model_name)
        model = models.init_decoder_weights(model, Y/Y.max(), c, is_unmixer=True, use_sivm=True)
        
        loader, _, _ = utils.create_dataloader(dataset, patch_size=H, dev=dev)

        epochs, lr = 400, 0.001

        optim_wrapper = dict(
            optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05),
            constructor='LayerDecayOptimizerConstructor_ViT',
            paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9, ))
        optimizer = build_optim_wrapper(model, optim_wrapper)

        for epoch in range(epochs):
            
            for Y, E, A in loader:
                optimizer.zero_grad()
                
                Y = utils.oneD_to_2d(Y).to(dev)

                Y, A, features = rsfm.extract_f(model_name, fm, Y, new_H, wavelengths, A)
                E_hat, A_hat, Y_hat = model(features)
                
                loss = model.loss(Y, Y_hat, A_hat, E_hat)
                
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)
        
        model.eval()
        _, A, features = rsfm.extract_f(model_name, fm, Y_init, new_H, wavelengths, A)
        E_hat1, A_hat1, Y_hat1 = model(features)

        sad, mse = utils.plot_results(E_hat1, A_hat1, A, E, normalise_E=True, normalise_A=True, return_results=True)
        mses[i_model, n] = float(mse)
        sads[i_model, n] = float(sad)

        print(f"Current MSE: {mse}, SAD: {sad}")

    return mses, sads, mses_cnn, sads_cnn

def main(args, dev):
    n_xp = 10

    datasets = ["samson", "urban""jasper", "apex"]

    # shape (n_datasets, n_models, n_xp)
    total_mses = torch.zeros(4, 3, n_xp)
    total_sads = torch.zeros(4, 3, n_xp)
    total_mses_cnn = torch.zeros(4, 3, n_xp)
    total_sads_cnn = torch.zeros(4, 3, n_xp)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float32).unsqueeze(0)
        E = torch.tensor(data["E"])
        A_flat = torch.tensor(data["A"]).unsqueeze(0)
        B, c, N = E.shape[0], E.shape[1], Y_flat.shape[1]

        Y = utils.oneD_to_2d(Y_flat)
        A = utils.oneD_to_2d(A_flat)
        H = Y.shape[-1]

        with open(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt", "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]

        mses = total_mses[i_dataset].cpu()
        sads = total_sads[i_dataset].cpu()
        mses_cnn = total_mses[i_dataset].cpu()
        sads_cnn = total_sads[i_dataset].cpu()

        for n in range(n_xp):
            print(f"------ Running {n+1}th experiment ------")
            mses, sads, mses_cnn, sads_cnn = run_one_xp(n, dataset, mses, sads, mses_cnn, sads_cnn, Y, wavelengths, B, c, H, dev)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

        total_mses[i_dataset] = mses
        total_sads[i_dataset] = sads
        total_mses_cnn[i_dataset] = mses_cnn
        total_sads_cnn[i_dataset] = sads_cnn
        
        mean_mses = torch.mean(mses, dim=1)
        mean_sads = torch.mean(sads, dim=1)
        std_mses = torch.std(mses, dim=1)
        std_sads = torch.std(sads, dim=1)

        mean_mses_cnn = torch.mean(mses_cnn, dim=1)
        mean_sads_cnn = torch.mean(sads_cnn, dim=1)
        std_mses_cnn = torch.std(mses_cnn, dim=1)
        std_sads_cnn = torch.std(sads_cnn, dim=1)

        model_list = ["DOFA", "SpecViT", "HyperFree", "HyperSIGMA"]
        for i_model, model in enumerate(model_list):
            wandb.log({f"{dataset}_{model}_BASIC MSE": mean_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD": mean_sads[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC MSE_std": std_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_std": std_sads[i_model]})

            wandb.log({f"{dataset}_{model}_BASIC MSE_cnn ": mean_mses_cnn [i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_cnn ": mean_sads_cnn [i_model]})
            wandb.log({f"{dataset}_{model}_BASIC MSE_std_cnn ": std_mses_cnn [i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_std_cnn ": std_sads_cnn [i_model]})

if __name__ == "__main__":

    logging.getLogger().setLevel(logging.WARNING) 

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_xp", default=5, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)

    run = wandb.init(project=f"FM_unmixing")
    
    print(f"Starting project on dev: {dev}")
    
    main(args, dev)