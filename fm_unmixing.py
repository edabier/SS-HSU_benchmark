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
import src.training.training as training
import src.models.models as models

import logging

def instantiate_models(Y, wavelengths, B, c, H):

    n_features = 1
    use_cls = False
    extend_cls = False

    model_list = {"DOFA": [], "SpecViT": [], "HyperFree": [], "HyperSIGMA": []}

    # DOFA
    if H < 224:
        Y_dofa = F.interpolate(Y, size=(224,224))
        H_dofa = 224
    else:
        Y_dofa = Y.clone()
        H_dofa = 224
    Y_dofa = Y_dofa[:,:,:H_dofa, :H_dofa]
    fm = rsfm.create_fm("DOFA", Y_dofa, n_features=n_features, use_cls=use_cls, extend_cls=extend_cls)
    features_dofa = rsfm.get_dofa_features(fm, Y_dofa, wavelengths)
    D_dofa = int(features_dofa.shape[0]/n_features)
    
    alpha = int(features_dofa.shape[1]**0.5)
    dofa_unmixer = rsfm.Unmixing_from_features(D=D_dofa, alpha=alpha, B=B, c=c, use_cls=use_cls)
    model_list["DOFA"].append([fm, dofa_unmixer])

    # SpecViT
    if H < 128:
        Y_specvit = F.interpolate(Y, size=(128,128))
        H_specvit = 128
    else:
        Y_specvit = Y.clone()
        H_specvit = 128
    Y_specvit = Y_specvit[:,:,:H_specvit, :H_specvit]
    fm = rsfm.create_fm("SpecViT", Y_specvit)
    features_specvit = rsfm.get_specvit_features(fm, Y_specvit, use_cls=use_cls)
    D_specvit = int(features_specvit.shape[0]/n_features)
    
    alpha = int(features_specvit.shape[1]**0.5)
    specvit_unmixer = rsfm.Unmixing_from_features(D=D_specvit, alpha=alpha, H=H_specvit, B=B, c=c, use_cls=use_cls)
    model_list["SpecViT"].append([fm, specvit_unmixer])

    # HyperFree
    patch_size_hyperfree = 16
    n_patches = H//patch_size_hyperfree
    if n_patches%2 != 0:
        H_hyperfree = (n_patches-1)*patch_size_hyperfree + patch_size_hyperfree-1
    else:
        H_hyperfree = H
    Y_hyperfree = Y[:, :, :H_hyperfree, :H_hyperfree]
    fm = rsfm.create_fm("HyperFree", Y_hyperfree, patch_size=patch_size_hyperfree)
    features_hyperfree = rsfm.get_hyperfree_features(fm, Y_hyperfree, wavelengths)
    D_hyperfree = features_hyperfree.shape[0]

    alpha = int(features_hyperfree.shape[1]**0.5)
    hyperfree_unmixer = rsfm.Unmixing_from_features(D=D_hyperfree, alpha=alpha, B=B, c=c, H=H_hyperfree, use_cls=use_cls)
    model_list["HyperFree"].append([fm, hyperfree_unmixer])

    # HyperSIGMA
    fm = rsfm.create_fm("HyperSIGMA", Y, c)
    features_hypersigma = rsfm.get_hypersigma_features(fm, Y)
    D = int(features_hypersigma.shape[1]/4)
    hypersigma_unmixer = rsfm.Unmixing_from_features(D=D, B=B, c=c, H=64, n_features=4, hypersig=True)
    model_list["HyperSIGMA"].append([fm, hypersigma_unmixer])
    H_hypersigma = H

    return model_list, H_hyperfree

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
            
    elif fm_name == "HyperFree":
        n_patches = H//16
        if n_patches%2 != 0:
            H = (n_patches-1)*16 + 16-1
        new_H = H
        Y = Y[:, :, :new_H, :new_H]
        return Y, new_H

    elif fm_name == "HyperSIGMA":
        new_H = H
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

    model_names = ["DOFA", "SpecViT", "HyperFree", "HyperSIGMA"]
    models_list, H_hyperfree = instantiate_models(Y_init, wavelengths, B, c, H)

    for i_model, model_name in enumerate(model_names): 

        print(f"Training {model_name}")

        fm = models_list[f"{model_name}"][0][0]
        model = models_list[f"{model_name}"][0][1]
        model.apply(model.weights_init)
        Y, new_H = transform_Y(Y_init, model_name)
        model = models.init_decoder_weights(model, Y/Y.max(), c, is_unmixer=True, use_sivm=True)
        
        if model_name == "HyperFree":
            loader, _, _ = utils.create_dataloader(dataset, patch_size=H_hyperfree, dev=dev)
        else:
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
                A = utils.oneD_to_2d(A).to(dev)
                E = E.to(dev)

                if model_name == "HyperSIGMA":
                    E_hat, A_hat, Y_hat = rsfm.unmix_full_image_hypersigma(Y, model, fm, c, use_bn=True)
                
                else:
                    Y, A, features = rsfm.extract_f(model_name, fm, Y, A, new_H, wavelengths, use_cls=False)
                    E_hat, A_hat, Y_hat = model(features)
                
                Yn = utils.sum_to_one(Y)
                W_ab, W_mse, W_tv_e, W_tv_a, W_e = 0.5, 0.2, 1e-2, 1e-7, 0#0.5, 1e-4, 1e-3, 1e-7, 1e-3
                loss, loss_sad, loss_ab, loss_tv_e, loss_tv_a, loss_mse, loss_norm_e = model.loss(Y, Y_hat, A_hat, E_hat, 1, W_ab, W_tv_e, W_tv_a, W_mse, W_e)

                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)
        
        if model_name == "HyperSIGMA":
            if Y_init.shape[-1] < new_H:
                Y_init = F.interpolate(Y_init, size=(new_H,new_H))
                A = F.interpolate(A, size=(new_H, new_H))
            E_hat1, A_hat1, Y_hat1 = rsfm.unmix_full_image_hypersigma(Y_init, model, fm, c, patch_size=64, use_bn=True)

        else:
            _, A, features = rsfm.extract_f(model_name, fm, Y_init, A, new_H, wavelengths)
            E_hat1, A_hat1, Y_hat1 = model(features)

        sad, mse = utils.plot_results(E_hat1, A_hat1, A, E, normalize_E=True, normalize_A=True, return_results=True)
        mses[i_model, n] = float(mse)
        sads[i_model, n] = float(sad)

        print(f"Current MSE: {mse}, SAD: {sad}")

        del E_hat1, A_hat1, Y_hat1, E_hat, A_hat, Y_hat, mse, sad, optimizer
        
        cnnaeu = rsfm.CNNAEU_with_decoder(B, c, model.decoder)

        epochs, lr = 500, 0.001
        optimizer = torch.optim.AdamW(cnnaeu.parameters(), lr=lr)

        for epoch in range(epochs):
            
            for Y_cnn, E_cnn, A_cnn in loader:
                optimizer.zero_grad()
                
                Y_cnn = Y_cnn.to(dev)
                E_hat_cnn, A_hat_cnn, Y_hat_cnn = cnnaeu(Y_cnn)
                loss = cnnaeu.loss(E_cnn, E_hat_cnn, A_cnn, A_hat_cnn, Y_cnn, Y_hat_cnn)
                
                loss.backward()
                optimizer.step()

        E_hat_cnn, A_hat_cnn, Y_hat_cnn = cnnaeu(Y.reshape(1, B, new_H**2))
        A_hat_cnn = utils.oneD_to_2d(A_hat_cnn)
        sad_cnn, mse_cnn = utils.plot_results(E_hat_cnn, A_hat_cnn, A, E, normalize_E=True, normalize_A=True, return_results=True)
        mses_cnn[i_model, n] = float(mse_cnn)
        sads_cnn[i_model, n] = float(sad_cnn)

        print(f"Current MSE_CNN: {mse_cnn}, SAD_CNN: {sad_cnn}")

        del E_hat_cnn, A_hat_cnn, Y_hat_cnn, fm, model, optimizer, mse_cnn, sad_cnn, loader

    del models_list
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    return mses, sads, mses_cnn, sads_cnn

def main(args, dev):
    n_xp = args.n_xp

    datasets = ["samson", "urban"]#"jasper", "apex", 

    # shape (n_datasets, n_models, n_xp)
    total_mses = torch.zeros(4, 4, n_xp)
    total_sads = torch.zeros(4, 4, n_xp)
    total_mses_cnn = torch.zeros(4, 4, n_xp)
    total_sads_cnn = torch.zeros(4, 4, n_xp)

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