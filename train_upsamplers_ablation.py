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

from src.utils import utils, plots, losses
from src.models import foundation_models as rsfm, models, unmixers as unmx

class FeaturesFusionUpsampler_ablation(nn.Module):
    def __init__(self, C, B, H):
        super().__init__()
        self.C = C
        self.B = B
        self.H = H

        # Step 1 : Extract features from high-res tensor (B channels)
        self.extract_hr = nn.Conv2d(B, C, kernel_size=1)  # Project B to C channels

        # Step 2 : Fuse upsampled low-res and high-res features
        self.fuse = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C),  # Concatenate and fuse
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x_hr):
        
        # Extract features from high-res tensor
        x_hr_feat = self.extract_hr(x_hr)  # (batch, C, H, H)

        # Fuse by concatenation
        x_fused = self.fuse(x_hr_feat)  # (batch, C, H, H)

        return x_fused
    
class UnmixingFromFeatures(nn.Module):
    def __init__(self, D, B, c, H=224, upsampler="Linear"):
        """
        Upsamples low res features then estimates A_hat
        
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features
            n_features (int): The size of the list of features in the case of several extracted features

        """
        super(UnmixingFromFeatures, self).__init__()
        self.D = D
        self.B = B
        self.c = c
        self.H = H
        self.upsampler = upsampler

        # Upsampling features
        if upsampler == "Linear":
            self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)
        elif upsampler == "Features_fusion":
            self.upsample = FeaturesFusionUpsampler_ablation(D, B, H)
        else:
            raise "Unknown upsampler, must be one of [Linear, Features_fusion]"
        self.spectral_regul = nn.Linear(D, D)

        # Upsampled features to abundances
        self.abundance_estimator = unmx.Abundance_estimator(self.D, self.c, 1)

        self.sum_to_one = unmx.Sum_to_one()
        self.decoder = unmx.Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09, hypersigma=False, return_losses=False):
        sad = losses.SADLoss()
        tv = losses.TVLoss(reduction="mean")
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        if hypersigma:
            loss_mse = W_mse * losses.hypersigma_mse(Y_gt, Y_hat)
        else:
            loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        """Feature regularisation"""

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        if return_losses:
            return loss, loss_sad, loss_ab, loss_tv_e, loss_mse
        else:
            return loss

    def get_abundances(self, Y):

        features_up = self.upsample(Y)
        features_up = features_up.view(
            1, self.D, self.H, self.H
        )
        features_up = utils.oneD_to_2d(self.spectral_regul(features_up.flatten(2).permute(0, 2, 1)).permute(0, 2, 1))

        features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
        A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, Y):
        A_hat = self.get_abundances(Y)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat

def instantiate_model(Y, wavelengths, model, version="v1", size="large"):
    fm, Y_init_fm, new_H = rsfm.create_fm(model, Y, version=version, size=size, path=global_path)
    _, features = rsfm.extract_f(fm, Y_init_fm, new_H, wavelengths)
    D = int(features.shape[0])
    alpha = int(features.shape[1]**0.5)

    return fm, Y_init_fm, new_H, D, alpha

def run_one_xp(i_dataset, upsampler, model, i_xp, dataset, mse_tensor, sad_tensor, Y_init, A_init, E_init, wavelengths, H, dev):

    Y_init = Y_init.to(dev)
    A_init = A_init.to(dev)
    E_init = E_init.to(dev)
    B, c = E_init.shape

    fm, Y_init_fm, new_H, D, alpha = instantiate_model(Y_init, wavelengths, model)
    n_train = 15
    E_hats, A_hats = torch.zeros(n_train, B, c), torch.zeros(n_train, c, new_H, new_H)

    for i in range(n_train): 
        
        print(f"Training {i+1}/{n_train}")

        model = UnmixingFromFeatures(D=D, B=B, c=c, H=new_H, upsampler=upsampler)

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

                Y_fm = rsfm.reshape_Y("DOFA", Y)

                E_hat, A_hat, Y_hat = model(Y_fm)

                loss = model.loss(Y_fm, Y_hat, A_hat, E_hat)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=1)
                optimizer.step()
                
                with torch.no_grad():
                    constraints = models.weightConstraint()
                    model.decoder.apply(constraints)
                    
        model.eval()
        
        with torch.no_grad():
        
            Y_fm, A_init_fm = rsfm.reshape_Y("DOFA", Y, new_H, A_init)
            E_hat, A_hat, _ = model(Y_fm)

            if not E_hat.isnan().any().item() and not A_hat.isnan().any().item():
                sad, _, mse = plots.compute_metrics_and_plot(E_hat, A_hat, A_init_fm, E_init, normalize_E=True, normalize_A=True, return_results=True, plot_E=False, plot_A=False)
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
        sad, _, mse = plots.compute_metrics_and_plot(E_hat_m, A_hat_m, A_init_fm, E_init, normalize_E=True, normalize_A=True, return_results=True, plot_E=False, plot_A=False)
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