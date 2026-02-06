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

import src.utils.utils as utils
import src.models.foundation_models as rsfm
import src.training.training as training
import src.models.models as models

def instantiate_models(Y, wavelengths, B, c, H):

    model_list = {"DOFA": [], "SpecViT": [], "HyperFree": [], "HyperSIGMA": []}

    # DOFA
    if H < 224:
        Y = F.interpolate(Y, size=(224,224))
        H_dofa = 224
    else:
        H_dofa = 224
    Y_dofa = Y[:,:,:H_dofa, :H_dofa]
    fm = rsfm.create_fm("DOFA", Y_dofa)
    features_dofa = rsfm.get_dofa_features(fm, Y_dofa, wavelengths)
    D_dofa = int(features_dofa.shape[0])
    
    alpha = int(features_dofa.shape[1]**0.5)
    dofa_unmixer = rsfm.Unmixing_from_features(D=D_dofa, alpha=alpha, B=B, c=c)
    model_list["DOFA"].append([fm, dofa_unmixer])

    # SpecViT
    if H < 128:
        Y = F.interpolate(Y, size=(128,128))
        H_specvit = 128
    else:
        H_specvit = 128
    Y_specvit = Y[:,:,:H_specvit, :H_specvit]
    fm = rsfm.create_fm("SpecViT", Y_specvit)
    features_specvit = rsfm.get_specvit_features(fm, Y_specvit)
    D_specvit = int(features_specvit.shape[0])
    
    alpha = int(features_specvit.shape[1]**0.5)
    specvit_unmixer = rsfm.Unmixing_from_features(D=D_specvit, alpha=alpha, H=H_specvit, B=B, c=c)
    model_list["SpecViT"].append([fm, specvit_unmixer])

    # HyperFree
    patch_size_hyperfree = 16
    n_patches = H//patch_size_hyperfree
    if n_patches%2 != 0:
        H_hyperfree = (n_patches-1)*patch_size_hyperfree + patch_size_hyperfree-1
    Y_hyperfree = Y[:, :, :H_hyperfree, :H_hyperfree]
    fm = rsfm.create_fm("HyperFree", Y_hyperfree, patch_size=patch_size_hyperfree)
    features_hyperfree = rsfm.get_hyperfree_features(fm, Y_hyperfree, wavelengths)
    D_hyperfree = features_hyperfree.shape[0]

    alpha = int(features_hyperfree.shape[1]**0.5)
    hyperfree_unmixer = rsfm.Unmixing_from_features(D=D_hyperfree, alpha=alpha, B=B, c=c, H=H_hyperfree)
    model_list["HyperFree"].append([fm, hyperfree_unmixer])

    # HyperSIGMA
    fm = rsfm.create_fm("HyperSIGMA", Y, c)
    features_hypersigma = rsfm.get_hypersigma_features(fm, Y)
    D = int(features_hypersigma.shape[1]/4)
    hypersigma_unmixer = rsfm.Unmixing_from_features(D=D, B=B, c=c, H=64, n_features=4, hypersig=True)
    model_list["HyperSIGMA"].append([fm, hypersigma_unmixer])

    return model_list, H_dofa, H_specvit, H_hyperfree

def run_one_xp(mses, sads, n, Y, wavelengths, B, c, H, loader, dev):
    """
    Instanciating models
    """
    model_list, H_dofa, H_specvit, H_hyperfree = instantiate_models(Y, wavelengths, B, c, H)
    model_names = ["DOFA", "SpecViT", "HyperFree", "HyperSIGMA"]

    epochs, lr = 400, 0.01

    """
    Instanciating trainers
    """
    for i_model, model_name in enumerate(model_names):

        model = model_list[f"{model_name}"][0][1]
        fm = model_list[f"{model_name}"][0][0]

        model.apply(model.weights_init)
        model = models.init_decoder_weights(model, Y, c, is_unmixer=True, use_sivm=True)

        print(f"Training {model_name}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        train_losses, sad_losses, ab_losses, tv_e_losses, tv_a_losses, mse_losses = [], [], [], [], [], []

        for epoch in range(epochs):
            train_loss, sad_loss, ab_loss, tv_e_loss, tv_a_loss, mse_loss = 0, 0, 0, 0, 0, 0
            
            for Y, E, A in loader:
                optimizer.zero_grad()
            
                Y = utils.oneD_to_2d(Y).to(dev)
                A = utils.oneD_to_2d(A).to(dev)
                E = E.to(dev)

                if model_name == "DOFA":
                    if Y.shape[-1] < H_dofa:
                        Y = F.interpolate(Y, size=(H_dofa,H_dofa))
                    Y = Y[:,:,:H_dofa, :H_dofa]
                    
                    if A.shape[-1] < H_dofa:
                        A = F.interpolate(A, size=(H_dofa,H_dofa))
                    A = A[:,:,:H_dofa, :H_dofa]

                    features = rsfm.get_dofa_features(fm, Y, wavelengths)
                    E_hat, A_hat, Y_hat = model(features)

                if model_name == "SpecViT":
                    if Y.shape[-1] < H_specvit:
                        Y = F.interpolate(Y, size=(H_specvit,H_specvit))
                    Y = Y[:,:,:H_specvit, :H_specvit]
                    
                    if A.shape[-1] < H_specvit:
                        A = F.interpolate(A, size=(H_specvit,H_specvit))
                    A = A[:,:,:H_specvit, :H_specvit]

                    features = rsfm.get_specvit_features(fm, Y)
                    E_hat, A_hat, Y_hat = model(features)

                elif model_name == "HyperFree":
                    Y = Y[:,:,:H_hyperfree,:H_hyperfree]
                    A = A[:,:,:H_hyperfree,:H_hyperfree]
                    features = rsfm.get_hyperfree_features(fm, Y, wavelengths)
                    E_hat, A_hat, Y_hat = model(features)

                elif model_name == "HyperSIGMA":
                    E_hat, A_hat, Y_hat = rsfm.unmix_full_image_hypersigma(Y, model, fm, c, use_bn=True)
                    features = None
                
                else:
                    pass
            
                W_sad, W_ab, W_tv_e, W_mse, W_tv_a = 1, 0.5, 1e-2, 0.2, 0
                loss, loss_sad, loss_ab, loss_tv_e, loss_tv_a, loss_mse = model.loss(Y, Y_hat, A_hat, E_hat, W_sad=W_sad, W_ab=W_ab, W_tv_e=W_tv_e, W_mse=W_mse, W_tv_a=W_tv_a)
                train_loss += loss.item()
                sad_loss += loss_sad.item()
                ab_loss += loss_ab.item()
                tv_e_loss += loss_tv_e.item()
                tv_a_loss += loss_tv_a.item()
                mse_loss += loss_mse.item()

                if epoch%(epochs//10)==0:
                    print(f"Epoch {epoch}: total = {format(loss, '.3f')}, SAD = {format(loss_sad, '.3f')}, MSE = {format(loss_mse, '.3f')}, Ab = {format(loss_ab, '.3f')}, TV_E = {format(tv_e_loss, '.3f')}, TV_A = {format(loss_tv_a, '.3f')}")
                
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)

                del E_hat, A_hat, Y_hat, loss, loss_sad, loss_ab, loss_tv_e, loss_mse, loss_tv_a
                torch.cuda.empty_cache()
                gc.collect()
                break

        if model_name == "DOFA" or model_name == "HyperFree" or model_name == "SpecViT":
            E_hat, A_hat, Y_hat = model(features)

        else:
            E_hat, A_hat, Y_hat = rsfm.unmix_full_image_hypersigma(Y, model, fm, c, patch_size=64, use_bn=True)
            
        total_sad, total_mse = utils.plot_results(E_hat, A_hat, A, E, normalize_E=True, normalize_A=True, return_results=True)
        mses[i_model, n] = total_mse 
        sads[i_model, n] = total_sad
        print(f"Current MSE: {total_mse}, SAD: {total_sad}")
                
        model.cpu()
        fm.cpu()
        del model, fm, features, E_hat, A_hat, Y_hat
        torch.cuda.empty_cache()
        gc.collect()
    
    del model_list
    torch.cuda.empty_cache()
    gc.collect()
    return mses, sads

def main(args, dev):
    n_xp = args.n_xp

    datasets = ["urban", "apex", "jasper", "samson"]

    # shape (n_datasets, n_models, n_xp)
    total_mses = torch.zeros(4, 4, n_xp, device=dev)
    total_sads = torch.zeros(4, 4, n_xp, device=dev)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float32)
        Y_flat = utils.normalize(Y_flat)
        E = torch.tensor(data["E"])
        A_flat = torch.tensor(data["A"])
        B, c, N = E.shape[0], E.shape[1], Y_flat.shape[1]

        Y = utils.oneD_to_2d(Y_flat)
        A = utils.oneD_to_2d(A_flat)
        H = Y.shape[-1]
        Y = Y.to(torch.float32)
        Y = Y.unsqueeze(0)

        with open(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}_wavelength", "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]

        loader, _, _ = utils.create_dataloader(dataset, patch_size=H, dev=dev)

        mses = total_mses[i_dataset]
        sads = total_sads[i_dataset]

        for n in range(n_xp):
            print(f"------ Running {n+1}th experiment ------")
            mses, sads = run_one_xp(mses, sads, n, Y, wavelengths, B, c, H, loader, dev)

        total_mses[i_dataset] = mses
        total_sads[i_dataset] = sads
        
        mean_mses = torch.mean(mses, dim=1)
        mean_sads = torch.mean(sads, dim=1)
        std_mses = torch.std(mses, dim=1)
        std_sads = torch.std(sads, dim=1)

        model_list = ["DOFA", "SpecViT", "HyperFree", "HyperSIGMA"]
        for i_model, model in enumerate(model_list):
            wandb.log({f"{dataset}_{model}_BASIC MSE": mean_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD": mean_sads[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC MSE_std": std_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_std": std_sads[i_model]})

if __name__ == "__main__":

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