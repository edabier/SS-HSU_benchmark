import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io as io
import sys
import optuna
import os
import gc

global_path = "/home/ids/edabier/HSU"
# global_path = "/home/edabier/Documents/Thèse/benchmark"
sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from mmengine.optim import build_optim_wrapper
from mmcv__custom import custom_layer_decay_optimizer_constructor

from src.utils import utils
from src.utils import plots
from src.models import models
from src.models import foundation_models as rsfm

def consistency_loss(A_hats, A_hat_m, pad, shift, similarity="mse"):
    H = A_hats[0].shape[-1]
    n_rows = 2*(pad-(shift-1))+1
    total_loss = 0.0

    for i in range(0, 2*pad + 1, shift):
        for j in range(0, 2*pad + 1, shift):

            i_model = n_rows*(i//shift) + j//shift
            A_hat_m_local = A_hat_m[:, :, i:i+H, j:j+H]

            if similarity == "mse":
                loss = F.mse_loss(A_hats[i_model], A_hat_m_local)
            elif similarity == "cosine":
                loss = 1 - F.cosine_similarity(A_hats[i_model], A_hat_m_local).mean()
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
        model = torch.compile(rsfm.Unmixing_from_features(D=D, alpha=alpha, H=new_H, B=B, c=c, n_features=1, use_cls=False, is_cnnaeu=False))
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

def objective(trial):    
    
    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)

    # Sample hyperparameters
    l_consistency = trial.suggest_float("l_consistency", 1e-7, 10, log=True)

    # List of datasets
    datasets = ["apex", "jasper", "samson", "urban"]
    all_sads, all_mses = [], []
    
    pad = 8
    shift = 8
    n_rows = 2*(pad-(shift-1))+1
    epochs=200

    ################ Testing on all datasets ################

    for dataset in datasets:
        
        print(f"Training {dataset}")
        
        # Load data
        data = io.loadmat(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float)
        A_flat = torch.tensor(data["A"], dtype=torch.float)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        B, c = E_init.shape
        Y_init = utils.oneD_to_2d(Y_flat)
        A_init = utils.oneD_to_2d(A_flat)
        H = Y_init.shape[-1]
        Y_init = Y_init.unsqueeze(0)
        A_init = A_init.unsqueeze(0)

        with open(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}_wavelength.txt", "r") as file:
            lines = file.readlines()
            wavelengths = [float(line.strip()) for line in lines if line.strip()]

        n_features = 1
        fm, Y_init_f, new_H = rsfm.create_fm("DOFA", Y_init, n_features=n_features, path=global_path)
        features = rsfm.get_dofa_features(fm, Y_init_f, wavelengths)
        D = int(features.shape[0]/n_features)
        alpha = int(features.shape[1]**0.5)
        
        Y_padded = F.pad(Y_init, pad=(pad, pad, pad, pad), mode="reflect")

        ################ Training models ################

        sads, mses = [], []
        for n in range(5):
            model_list, optimizers = instantiate_models(pad, shift, Y_init, D, alpha, H, B, c, dev)
            total_loss = 0
            A_hats = []

            for epoch in range(epochs):

                total_loss = 0
                A_hats = []
                
                A_hat_padded = torch.zeros(1, c, H+2*pad, H+2*pad)
                weights = torch.zeros_like(A_hat_padded)
                E_hats = torch.zeros((2*pad+1)**2, B, c)

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

                        E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)
                        A_hat = A_hat.unsqueeze(0)
                        A_hats.append(A_hat)
                        E_hats[i_model] = E_hat

                        A_hat_padded[:, :, i:i+H, j:j+H] += A_hat
                        weights[:, :, i:i+H, j:j+H] += 1
                        
                        loss = model.loss(Y_crop, Y_hat, A_hat, E_hat)
                        total_loss += loss
                
                A_hat_padded /= weights
                E_hat_m = torch.mean(E_hats, dim=0)    
                consistency = consistency_loss(A_hats, A_hat_padded, pad, shift)
                total_loss += l_consistency * consistency
                
                total_loss.backward()

                for i_model, model in enumerate(model_list):
                    optimizer = optimizers[i_model]    
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()

                    with torch.no_grad():
                        constraints = models.weightConstraint()
                        model.decoder.apply(constraints)
                
                if epoch%(epochs//10)==0:
                    print(f"Current epoch: {epoch}")
                    
            del optimizers, total_loss, E_hat_m, A_hat_padded, E_hats, A_hats, E_hat, A_hat, weights, loss, consistency
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            ################ Testing models ################

            with torch.no_grad():
                A_hat_padded = torch.zeros(1, c, H+2*pad,H+2*pad)
                weights = torch.zeros_like(A_hat_padded)
                E_hats = torch.zeros((2*pad+1)**2, B, c)

                for i in range(0, 2*pad + 1, shift):
                    for j in range(0, 2*pad + 1, shift):

                        i_model = n_rows*(i//shift) + j//shift
                        model = model_list[i_model]
                        model.eval()
                        Y_crop = Y_padded[:, :, i:i+H, j:j+H]
                        Y_crop = Y_crop.to(dev)
                        _, features = rsfm.extract_f(fm, Y_crop, H, wavelengths, use_cls=False)
                        E_hat, A_hat, Y_hat = model(features)
                        
                        E_hat, A_hat, _ = utils.order_endmembers(E_init, E_hat, A_hat)
                        A_hat = A_hat.unsqueeze(0)
                        E_hats[i_model] = E_hat
                        A_hat_padded[:, :, i:i+H, j:j+H] += A_hat
                        weights[:, :, i:i+H, j:j+H] += 1
                
                A_hat_padded /= weights
                E_hat_m = torch.mean(E_hats, dim=0) 

            sad, _, mse = plots.compute_metrics_and_plot(E_hat_m, A_hat_padded[:,:,pad*2:-1-2*pad, pad*2:-1-2*pad], A_init[:, :,pad:-1-pad, pad:-1-pad], E_init, return_results=True, plot_A=False, plot_E=False)
            # all_sads.append(sad)
            # all_mses.append(mse)
            sads.append(sad)
            mses.append(mse)
        
            del model_list, E_hat, A_hat, E_hats, A_hat_padded, weights, E_hat_m, sad, mse
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            
        all_sads.extend(sads)
        all_mses.extend(mses)
        del A_init, E_init, Y_init, features, alpha, D, fm, wavelengths, data, sads, mses
        
    avg_sad = torch.mean(torch.tensor(all_sads))
    avg_mse = torch.mean(torch.tensor(all_mses))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    return avg_sad, avg_mse

if __name__ == "__main__":

    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
        print(f"Using device: {dev}")
    
    else:
        dev = "cpu"
        print(f"Using device: {dev}")

    study = optuna.create_study(directions=["minimize", "minimize"])  # Minimize both SAD and MSE
    study.optimize(objective, n_trials=50)

    # Print the best trials on the Pareto front
    print("Pareto-optimal trials:")
    for trial in study.best_trials:
        print(f"  SAD: {trial.values[0]}, MSE: {trial.values[1]}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")