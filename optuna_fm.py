import optuna
import torch
import torch.nn as nn
import numpy as np
import scipy.io as io
import sys
import os

global_path = "/home/ids/edabier/HSU"
sys.path.append(f"{global_path}/HyperSIGMA/HyperspectralUnmixing")
from mmengine.optim import build_optim_wrapper
from mmcv__custom import custom_layer_decay_optimizer_constructor

import src.utils.utils as utils
import src.utils.plots as plots
import src.models.foundation_models as rsfm
import src.models.models as models

def objective(trial):    
    
    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)

    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-3, 1, log=True)
    W_ab = trial.suggest_float("W_ab", 1e-3, 1, log=True)
    W_mse = trial.suggest_float("W_mse", 1e-4, 1e-1, log=True)
    W_tv_e = trial.suggest_float("W_tv_e", 1e-6, 1e-3, log=True)
    W_tv_a = trial.suggest_float("W_tv_a", 1e-9, 1e-6, log=True)
    W_e = trial.suggest_float("W_e", 1e-11, 1e-5, log=True)

    # List of datasets
    datasets = ["apex", "jasper", "samson", "urban"]
    all_sads, all_mses = [], []

    for dataset in datasets:
        # Load data
        data = io.loadmat(f"{global_path}/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_flat = torch.tensor(data["Y"], dtype=torch.float)
        A_flat = torch.tensor(data["A"], dtype=torch.float)
        E_init = torch.tensor(data["E"], dtype=torch.float)
        B, c, N = E_init.shape[0], E_init.shape[1], Y_flat.shape[1]
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
        loader, _, _ = utils.create_dataloader(dataset, patch_size=H, dev=dev, path=global_path)

        # Run 5 iterations
        sads, mses = [], []
        for n in range(5):
            model = rsfm.Unmixing_from_features(D=D, alpha=alpha, H=new_H, B=B, c=c, n_features=n_features, use_cls=False, is_cnnaeu=False)
            model = model.to(dev)

            model.apply(model.weights_init)
            model = models.init_decoder_weights(model, Y_init/Y_init.max(), c, is_unmixer=True)
            optim_wrapper = dict(
                optimizer=dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay),
                constructor='LayerDecayOptimizerConstructor_ViT',
                paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
            optimizer = build_optim_wrapper(model, optim_wrapper)

            epochs = 200
            for epoch in range(epochs):
                for Y, E, A in loader:
                    optimizer.zero_grad()
                    Y = utils.oneD_to_2d(Y).to(dev)
                    A = utils.oneD_to_2d(A).to(dev)
                    E = E.to(dev)
                    Y, A, features = rsfm.extract_f(fm, Y, new_H, wavelengths, A, False)
                    E_hat, A_hat, Y_hat = model(features)
                    loss, _, _, _, _, _, _ = model.loss(Y, Y_hat, A_hat, E_hat, 1, W_ab, W_tv_e, W_tv_a, W_mse, W_e)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
                    optimizer.step()
                    with torch.no_grad():
                        constraints = models.weightConstraint()
                        model.decoder.apply(constraints)
                    break

            model.eval()
            with torch.no_grad():
                _, A_init, features = rsfm.extract_f(fm, Y_init, new_H, wavelengths, A, False)
                E_hat1, A_hat1, _ = model(features)

            sad, _, mse = plots.compute_metrics_and_plot(E_hat1, A_hat1, A_init, E, normalize_E=True, normalize_A=True, return_results=True, plot_A=False, plot_E=False)
            sads.append(sad.detach().cpu())
            mses.append(mse.detach().cpu())

        all_sads.extend(sads)
        all_mses.extend(mses)

    avg_sad = np.mean(all_sads)
    avg_mse = np.mean(all_mses)
    return avg_sad, avg_mse

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)
        print(f"Starting optuna on dev: {dev}")

    study = optuna.create_study(directions=["minimize", "minimize"])  # Minimize both SAD and MSE
    study.optimize(objective, n_trials=150)

    # Print the best trials on the Pareto front
    print("Pareto-optimal trials:")
    for trial in study.best_trials:
        print(f"  SAD: {trial.values[0]}, MSE: {trial.values[1]}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")