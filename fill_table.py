import torch
import scipy.io as io
import argparse
import matplotlib.pyplot as plt
import wandb
import os
import sys
import gc

import src.utils.utils as utils
import src.models.models as models
import src.training.self_supervision as ssl
import src.training.training as training
import src.utils.extractor as extractor

def run_one_xp(mses, sads, n, B, c, N, H, loader, dataset, args, dev):
    """
    Instanciating models
    """
    model_list = []

    # CNNAEU
    Y_loader, e_gt, a_gt = next(iter(loader))[0][0], next(iter(loader))[1][0], next(iter(loader))[2][0]
    cnnaeu = models.CNNAEU(B=B, c =c)
    cnnaeu = models.init_decoder_weights(cnnaeu, Y_loader, c, 11)
    model_list.append(cnnaeu)
    
    # Deep Trans
    im_size = (H//5)*5
    deep_trans = models.DeepTrans(B=B, c=c, im_size=im_size)
    deep_trans = models.init_decoder_weights(deep_trans, Y_loader, c)
    model_list.append(deep_trans)
    
    # UnDIP
    undip = models.UnDIP(B=B, c=c)
    model_list.append(undip)
    
    # NALMU
    nalmu = models.NALMU(T=25, B=B, c=c, N=N)
    nalmu = nalmu.to(dev)
    model_list.append(nalmu)

    # RALMU
    ralmu = models.RALMU(T=25, B=B, c=c, im_size=H)
    ralmu = ralmu.to(dev)
    model_list.append(ralmu)

    """
    Instanciating trainers
    """
    for i_model, model in enumerate(model_list):
        model_name = model.__class__.__name__
        model = model.to(dev)

        if model_name == "NALMU" or model_name == "RALMU":
            model_name += str(model.T)

        print(f"Training {model_name}")

        if model_name == "UnDIP" or "NALMU" in model_name or "RALMU" in model_name:
            e_hat, a_hat, train_losses = training.train(model, loader, has_decoder=False, epochs=args.epochs, lr=args.lr, dev=dev)
        else:
            e_hat, a_hat, train_losses = training.train(model, loader, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr, dev=dev)

        fig = plt.figure()
        plt.plot(train_losses)
        wandb.log({f"{dataset}_{model_name}_train_loss": wandb.Image(fig)})
        plt.close(fig)
        del e_hat, a_hat
        torch.cuda.empty_cache()

        model.load_state_dict(torch.load(f"/home/ids/edabier/HSU/SS-HSU_benchmark/models/{model_name}_BASIC_{dataset}_lr_{args.lr}.pt")["model_state_dict"], strict=False)
        
        batch = next(iter(loader))
        Y, e_gt, a_gt = batch[0][0], batch[1][0], batch[2][0]

        with torch.no_grad():
            if model_name == "DeepTrans":
                Y, a_gt = utils.crop_patch_image(Y, args.patch_size, a_gt)

            if "NALMU" in model_name or "RALMU" in model_name:
                A_init_disp = next(iter(loader))[2].to(torch.float32)
                E_init_disp = torch.ones(next(iter(loader))[1].size(),dtype=torch.float32)
                A_init = torch.ones_like(A_init_disp)
                E_init = torch.ones_like(E_init_disp)
                e_hat, a_hat, y_hat = model.forward(Y, E_init=E_init, A_init=A_init)
            else:
                e_hat, a_hat, y_hat = model.forward(Y)

        if e_hat.dim() == 3:
            e_hat = e_hat.squeeze(0)
        if a_hat.dim() == 3:
            a_hat = a_hat.squeeze(0)
        
        a_hat = utils.oneD_to_2d(a_hat)
        a_gt = utils.oneD_to_2d(a_gt)   
        
        mse, sad = utils.compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=f"{model_name}_BASIC_{dataset}", use_wandb=True)
        mses[i_model, n] = mse
        sads[i_model, n] = sad
        print(f"Current MSE: {mse}, SAD: {sad}")
        
        del e_hat, a_hat, y_hat
        del model
        torch.cuda.empty_cache()
    
    return mses, sads

def main(args, dev):
    n_xp = args.n_xp

    datasets = ["urban", "apex", "jasper", "samson"]

    # shape (n_models, n_xp)
    mses = torch.zeros(4, n_xp, device=dev)
    sads = torch.zeros(4, n_xp, device=dev)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_init = torch.tensor(data["Y"], device=dev)
        Y_init = Y_init.to(torch.float32)
        E = torch.tensor(data["E"])
        B, c, N = E.shape[0], E.shape[1], Y_init.shape[1]

        Y_init = utils.oneD_to_2d(Y_init)
        H, W = Y_init.shape[1], Y_init.shape[2]

        loader, _, _ = utils.get_dataloader(dataset, patch_size=64, batch_size=1)

        for n in range(n_xp):
            print(f"------ Running {n+1}th experiment ------")
            mses, sads = run_one_xp(mses, sads, n, B, c, N, H, loader, dataset, args, dev)

        mean_mses = torch.mean(mses, dim=1)
        mean_sads = torch.mean(sads, dim=1)
        std_mses = torch.std(mses, dim=1)
        std_sads = torch.std(sads, dim=1)

        model_list = ["NALMU", "RALMU", "CNNAEU", "DeepTrans", "UnDIP"]
        for i_model, model in enumerate(model_list):
            wandb.log({f"{dataset}_{model}_BASIC MSE": mean_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD": mean_sads[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC MSE_std": std_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_std": std_sads[i_model]})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--patch_size", default=5, type=int)
    parser.add_argument("--n_xp", default=5, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("GPUs seen by torch:", torch.cuda.device_count())
        sys.exit("FATAL: CUDA not available — aborting")
    else:
        dev = "cuda:0"
        torch.set_default_device(dev)

    run = wandb.init(project=f"Fill_table",
                     config={
                        "learning_rate": args.lr,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                    },)
    
    print(f"Starting project on dev: {dev}")
    
    main(args, dev)