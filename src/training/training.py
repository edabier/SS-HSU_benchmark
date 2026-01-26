import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
import sys
import os
import scipy.io as io
import argparse
import wandb

sys.path.append("/home/ids/edabier/HSU/DOFA")
from dofa_v1 import vit_base_patch16

import src.utils.extractor as extractor
import src.models.foundation_models as rsfm
import src.utils.utils as utils
import src.models.models as models

directory = "/home/ids/edabier/HSU/SS-HSU_benchmark/models"
# directory = "models/"

def train(model, dataloader, patch_size=None, has_decoder=True, epochs=320, lr=0.003, dev="cpu"):
    model_name = model.__class__.__name__
    if model_name == "NALMU" or model_name == "RALMU":
        model_name += str(model.T)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses = []
    for epoch in range(epochs):
        
        train_loss = 0
        
        for Y, E, A in dataloader:
            optimizer.zero_grad()
            
            Y = Y.to(dev)
            E = E.to(dev)
            A = A.to(dev)

            if model_name == "DeepTrans":
                Y, A = utils.crop_patch_image(Y, patch_size, A)
            
            if "NALMU" in model_name or "RALMU" in model_name:
                A_init_disp = A.to(torch.float32)
                E_init_disp = torch.ones(E.size(),dtype=torch.float32)
                A_init = torch.ones_like(A_init_disp)
                E_init = torch.ones_like(E_init_disp)
                e_hat, a_hat, y_hat = model(Y, E_init=E_init, A_init=A_init)
            else:
                e_hat, a_hat, y_hat = model(Y)
            
            loss = model.loss(E, e_hat, A, a_hat, Y, y_hat)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            if has_decoder:
                with torch.no_grad():
                    model.decoder.apply(models.weightConstraint())
    
        train_loss /= len(dataloader)
        train_losses.append(train_loss)
        
        try:
            dataset_name = dataloader.dataset.dataset_name
        except:
            dataset_name = dataloader.dataset.dataset.dataset_name

        # Save checkpoint
        utils.save_model(model, optimizer, directory=directory, name=f"BASIC_{dataset_name}", epoch=epoch)
            
    return e_hat, a_hat, train_losses

def evaluate_model(params, model, dofa, wavelengths, loader, dev):
    epochs, lambda_1, lambda_2, lambda_3, lr = params
    epochs = int(epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    for epoch in range(epochs):
        train_loss = 0

        for Y, E, A in loader:
            optimizer.zero_grad()

            Y = utils.oneD_to_2d(Y).to(dev)
            if Y.shape[-1] < 224:
                Y = F.interpolate(Y, size=(224,224))
            E = E.to(dev)
            A = utils.oneD_to_2d(A).to(dev)
            if A.shape[-1] < 224:
                A = F.interpolate(A, size=(224,224))

            features = dofa.forward_features(Y, wavelengths)
            E_hat, A_hat, Y_hat = model(features)
            loss = model.loss(Y, Y_hat, A_hat, E_hat, W_sad=1, W_ab=lambda_1, W_tv=lambda_2, W_mse=lambda_3)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                constraints = models.weightConstraint()
                model.decoder.apply(constraints)
            break
    
    with torch.no_grad():
        Y, E, A = next(iter(loader))
        Y = utils.oneD_to_2d(Y).to(dev)
        if Y.shape[-1] < 224:
            Y = F.interpolate(Y, size=(224,224))
        E = E.to(dev)
        A = utils.oneD_to_2d(A).to(dev)
        if A.shape[-1] < 224:
            A = F.interpolate(A, size=(224,224))

        features = dofa.forward_features(Y, wavelengths)
        E_hat, A_hat, _ = model(features)
        metric_A, metric_E = utils.compute_metrics(E, A, E_hat, A_hat)
    metric = metric_A + metric_E
    return metric.item()

def grid_search(param_space, model, dofa, wavelengths, loader, dev):
    grid = ParameterGrid(param_space)
    results = []

    for params in grid:
        metric = evaluate_model(list(params.values()), model, dofa, wavelengths, loader, dev)
        results.append((params, metric))
    
    return results

def bayesian_optimization(model, dofa, wavelengths, loader, dev, n_calls=50):
    dimensions = [
        Integer(1000, 4000, name='epochs'),
        Real(1e-6, 1, prior='log-uniform', name='lambda_1'),
        Real(1e-6, 1e-2, prior='log-uniform', name='lambda_2'),
        Real(1e-6, 1e-4, prior='log-uniform', name='lambda_3'),
        Real(1e-5, 1e-1, prior='log-uniform', name='lr'),
    ]

    @use_named_args(dimensions=dimensions)
    def objective(lambda_1, lambda_2, lambda_3, lr, epochs):
        return evaluate_model([epochs, lambda_1, lambda_2, lambda_3, lr], model, dofa, wavelengths, loader, dev)

    result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=42)
    return result

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="urban", type=str)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--n_features", default=1, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)

    run = wandb.init(project=f"FM_unmixing",
                     config={
                        "dataset": args.dataset,
                        "n_features": args.n_features,
                    },)
    
    dataset     = args.dataset
    n_features  = args.n_features
    patch_size  = args.patch_size
    data    = io.loadmat(f"datasets/{dataset}.mat")
    Y_flat  = torch.tensor(data["Y"], dtype=torch.float32)
    Y_flat  = utils.normalize(Y_flat)
    E = torch.tensor(data["E"])
    B, c, N = E.shape[0], E.shape[1], Y_flat.shape[1]

    Y = utils.oneD_to_2d(Y_flat)
    B = Y.shape[0]
    H = Y.shape[-1]
    Y = Y.to(torch.float32)
    Y = Y.unsqueeze(0)
    if H < 224:
        Y = F.interpolate(Y, size=(224,224))
    else:
        H = 224

    with open(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}_wavelength", "r") as file:
        lines = file.readlines()
        wavelengths = [float(line.strip()) for line in lines if line.strip()]
        
    check_point = torch.load('/home/ids/edabier/HSU/DOFA/checkpoints/DOFA_ViT_base_e100.pth', map_location=dev)
    dofa = vit_base_patch16(n_features=n_features)
    dofa.load_state_dict(check_point, strict=False)

    Y = Y[:,:,:224, :224]
    Y_flat = Y.reshape(1, B, 224**2)
    D = 768
    model = rsfm.Unmixing_from_features(D=D, p=14, B=B, c=c, n_features=n_features)
    model = models.init_decoder_weights(model, Y, c, is_unmixer=True, use_sivm=True)

    loader, _, _ = utils.create_dataloader(dataset, patch_size=H, dev=dev)
    param_space = {
        'lambda_1': np.logspace(-6, 0, 7),  # ab [1e-6, 1]
        'lambda_2': np.logspace(-6, -2, 5),  # tv [1e-6, 1e-2]
        'lambda_3': np.logspace(-6, -4, 3),  # mse [1e-6, 1e-4]
        'lr': np.logspace(-5, -1, 5),        # lr [1e-5, 1e-1]
        'epochs': [1000, 2000, 4000],        # epochs
    }

    results = bayesian_optimization(model, dofa, wavelengths, loader, dev, n_calls=10)
    # results = grid_search(param_space, model, dofa, wavelengths, loader, dev)

    # for i in range(len(results)):
    #     wandb.log({f"{dataset}_{model}_BASIC MSE": mse})