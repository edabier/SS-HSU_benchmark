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
import src.utils.extractor as extractor

def run_one_xp(mses, sads, n, Y_init, B, c, N, H, loader, dataset, args, dev):
    """
    Instanciating models
    """
    model_list = []

    # # CNNAEU
    # Y_loader, e_gt, a_gt = next(iter(loader))[0][0], next(iter(loader))[1][0], next(iter(loader))[2][0]
    # cnnaeu = models.CNNAEU(B=B, c =c)
    # cnnaeu = models.init_decoder_weights(cnnaeu, Y_loader, c, 11)
    # model_list.append(cnnaeu)
    
    # # Deep Trans
    # Y_trans = Y_init[:,:(Y_init.shape[1]//args.patch_size)*args.patch_size,:(Y_init.shape[1]//args.patch_size)*args.patch_size]
    # deep_trans = models.DeepTrans(B=B, c=c, im_size=Y_trans.shape[1], dim=200)
    # deep_trans = models.init_decoder_weights(deep_trans, Y_loader, c)
    # model_list.append(deep_trans)
    
    # # UnDIP
    # undip = models.UnDIP(B=B, c=c)
    # model_list.append(undip)
    
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

        if model_name == "NALMU" or model_name == "RALMU":
            model_name += str(model.T)

        print(f"Training {model_name}")

        if model_name == "UnDIP" or "NALMU" in model_name or "RALMU" in model_name:
            e_hat, a_hat, train_losses = ssl.train(model, loader, has_decoder=False, epochs=args.epochs, lr=args.lr, dev=dev)
        else:
            e_hat, a_hat, train_losses = ssl.train(model, loader, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr, dev=dev)

        fig = plt.figure()
        plt.plot(train_losses)
        wandb.log({f"{dataset}_{model_name}_train_loss": wandb.Image(fig)})
        plt.close(fig)
        del e_hat, a_hat
        torch.cuda.empty_cache()

        model.load_state_dict(torch.load(f"/home/ids/edabier/HSU/SS-HSU_benchmark/models/{model_name}_Basic_{dataset}_lr_{args.lr}.pt")["model_state_dict"], strict=False)
        
        batch = next(iter(loader))
        Y, e_gt, a_gt = batch[0][0], batch[1][0], batch[2][0]
    
        A_init_disp = next(iter(loader))[2].to(torch.float32)
        E_init_disp = torch.ones(next(iter(loader))[1].size(),dtype=torch.float32)
        A_init = torch.ones_like(A_init_disp)
        E_init = torch.ones_like(E_init_disp)

        with torch.no_grad():
            if model_name == "DeepTrans":
                Y, a_gt = utils.crop_patch_image(Y, args.patch_size, a_gt)

            if "NALMU" in model_name or "RALMU" in model_name:
                e_hat, a_hat, y_hat = model.forward(Y, E_init=E_init, A_init=A_init)
            else:
                e_hat, a_hat, y_hat = model.forward(Y)

        if e_hat.dim() == 3:
            e_hat = e_hat.squeeze(0)
        if a_hat.dim() == 3:
            a_hat = a_hat.squeeze(0)
        
        a_hat, _ = utils.oneD_to_2d(a_hat)
        a_gt, _ = utils.oneD_to_2d(a_gt)   
        
        mse, sad = utils.compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=f"{model_name}_BASIC_{dataset}", use_wandb=True)
        mses[i_model, n] = torch.tensor(mse[-1]).item()
        sads[i_model, n] = torch.tensor(sad[-1]).item()
        
        del e_hat, a_hat, y_hat
        del A_init, E_init
        del model
        torch.cuda.empty_cache()
    
    return mses, sads

def main(args, dev):
    n_xp = args.n_xp

    datasets = ["urban", "apex", "jasper", "samson"]

    # shape (n_models, n_xp)
    mses = torch.zeros(3, n_xp, device=dev)
    sads = torch.zeros(3, n_xp, device=dev)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_init = torch.tensor(data["Y"], device=dev)
        Y_init = Y_init.to(torch.float32)
        E = torch.tensor(data["E"])
        B, c, N = E.shape[0], E.shape[1], Y_init.shape[1]

        Y_init, _ = utils.oneD_to_2d(Y_init)
        H, W = Y_init.shape[1], Y_init.shape[2]

        loader, _, _ = utils.create_dataloader(dataset, dev=dev, batch_size=args.batch_size)

        for n in range(n_xp):
            print(f"------ Running {n+1}th experiment ------")
            mses, sads = run_one_xp(mses, sads, n, Y_init, B, c, N, H, loader, dataset, args, dev)
        
        batch = next(iter(loader))
        Y, e_gt, a_gt = batch[0][0], batch[1][0], batch[2][0]

        e_hat, a_hat = extractor.unmix(Y, c, use_sivm=True)

        if e_hat.dim() == 3:
            e_hat = e_hat.squeeze(0)
        if a_hat.dim() == 3:
            a_hat = a_hat.squeeze(0)
        
        a_hat, _ = utils.oneD_to_2d(a_hat)
        a_gt, _ = utils.oneD_to_2d(a_gt)    
        
        mse, sad = utils.compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=f"SiVM+FCLS_{dataset}", use_wandb=True)
        mses[2, 0] = torch.tensor(mse[-1]).item()
        sads[2, 0] = torch.tensor(sad[-1]).item()
        
        mean_mses = torch.mean(mses, dim=1)
        std_mses = torch.std(mses, dim=1)
        mean_sads = torch.mean(sads, dim=1)
        std_sads = torch.std(sads, dim=1)

        model_list = ["NALMU", "RALMU", "SiVM+FCLS"] #["CNNAEU", "DeepTrans", "UnDIP"]
        for i_model, model in enumerate(model_list):
            wandb.log({f"{dataset}_{model}_BASIC MSE": mean_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD": mean_sads[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC MSE_std": std_mses[i_model]})
            wandb.log({f"{dataset}_{model}_BASIC SAD_std": std_sads[i_model]})

def run_one_xp_trainers(mses, sads, i_dataset, n, Y_init, B, c, N, H, loader, dataset, args, dev):
    """
    Instanciating models
    """
    model_list = []

    # # MLP
    # mlpae = models.MLP_AE(B=B, c=c)
    # model_list.append(mlpae)

    # CNN + linear decoder
    # cnnlinear = models.CNNAE_linear(B=B, c=c, patch_size=args.patch_size)
    # model_list.append(cnnlinear)

    # # CNNAEU
    # Y_loader, e_gt, a_gt = next(iter(loader))[0][0], next(iter(loader))[1][0], next(iter(loader))[2][0]
    # cnnaeu = models.CNNAEU(B=B, c =c)
    # cnnaeu = models.init_decoder_weights(cnnaeu, Y_loader, c, 11)
    # model_list.append(cnnaeu)
    
    # Deep Trans
    # Y_trans = Y_init[:,:(Y_init.shape[1]//args.patch_size)*args.patch_size,:(Y_init.shape[1]//args.patch_size)*args.patch_size]
    # deep_trans = models.DeepTrans(B=B, c=c, im_size=Y_trans.shape[1], dim=200)
    # deep_trans = models.init_decoder_weights(deep_trans, Y_loader, c)
    # model_list.append(deep_trans)
    
    # NALMU
    nalmu = models.NALMU(B=B, c=c, N=N)
    nalmu = nalmu.to(dev)
    model_list.append(nalmu)

    # RALMU
    ralmu = models.RALMU(B=B, c=c, im_size=H)
    ralmu = ralmu.to(dev)
    model_list.append(ralmu)

    """
    Instanciating trainers
    """
    for i_model, model in enumerate(model_list):

        print(f"Training {model.__class__.__name__}")

        trainers = []

        # Supervised training
        supervised_trainer = ssl.SupervisedTrainer(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
        supervised_trainer_trans = ssl.SupervisedTrainer(model, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr)
        trainers.append([supervised_trainer, supervised_trainer_trans])

        # Reconstruction Error
        re_trainer = ssl.ReconstructionError(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
        re_trainer_trans = ssl.ReconstructionError(model, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr)
        trainers.append([re_trainer, re_trainer_trans])

        # DIP
        dip_trainer = ssl.DIP(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
        dip_trainer_trans = ssl.DIP(model, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr)
        trainers.append([dip_trainer, dip_trainer_trans])

        # Two stages
        twos_trainer = ssl.TwoStagesNet(model, B, has_decoder=False, epochs=args.epochs, lr=args.lr)
        twos_trainer_trans = ssl.TwoStagesNet(model, B, patch_size=args.patch_size, has_decoder=True, epochs=args.epochs, lr=args.lr)
        trainers.append([twos_trainer, twos_trainer_trans])

        for i_trainer, trainer_ in enumerate(trainers):
            model_name = model.__class__.__name__
            
            if model_name == "Transformer_AE":
                trainer = trainer_[1]
            else:
                trainer = trainer_[0]
            trainer_name = trainer.__class__.__name__

            print(f"Starting {trainer_name} startegy")

            e_hat, a_hat, train_losses = trainer.train(loader, dev)
            
            fig = plt.figure()
            plt.plot(train_losses)
            wandb.log({f"{dataset}_{model_name}_{trainer_name}_train_loss": wandb.Image(fig)})
            plt.close(fig)
            del e_hat, a_hat
            del trainer
            torch.cuda.empty_cache()

            model.load_state_dict(torch.load(f"/home/ids/edabier/HSU/SS-HSU_benchmark/models/{model_name}_{trainer_name}_{dataset}_lr_{args.lr}.pt")["model_state_dict"], strict=False)
            
            batch = next(iter(loader))
            Y, e_gt, a_gt = batch[0][0], batch[1][0], batch[2][0]
            
            A_init_disp = next(iter(loader))[2].to(torch.float32)
            E_init_disp = torch.ones(next(iter(loader))[1].size(),dtype=torch.float32)
            A_init = torch.ones_like(A_init_disp)
            E_init = torch.ones_like(E_init_disp)

            with torch.no_grad():
                if model_name == "DeepTrans":
                    Y, a_gt = utils.crop_patch_image(Y, args.patch_size, a_gt)
                e_hat, a_hat, y_hat = model.forward(Y, E_init=E_init, A_init=A_init)

            if e_hat.dim() == 3:
                e_hat = e_hat.squeeze(0)
            
            a_hat = a_hat.reshape(a_hat.shape[1], int(a_hat.shape[2]**0.5), int(a_hat.shape[2]**0.5))
            a_gt = a_gt.reshape(a_gt.shape[0], int(a_gt.shape[1]**0.5), int(a_gt.shape[1]**0.5))    
            
            mse, sad = utils.compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=f"{model_name}_{trainer_name}_{dataset}", use_wandb=True)
            mses[i_model, i_trainer, n] = torch.tensor(mse[-1]).item()
            sads[i_model, i_trainer, n] = torch.tensor(sad[-1]).item()
            
            del e_hat, a_hat, y_hat
            del A_init, E_init
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    
    return mses, sads

def main_trainers(args, dev):
    n_xp = args.n_xp

    datasets = ["urban", "apex", "jasper", "samson"]

    # shape (n_models, n_trainers, n_xp)
    mses = torch.zeros(2, 4, n_xp, device=dev)
    sads = torch.zeros(2, 4, n_xp, device=dev)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")
        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_init = torch.tensor(data["Y"], device=dev)
        Y_init = Y_init.to(torch.float32)
        E = torch.tensor(data["E"])
        B, c, N = E.shape[0], E.shape[1], Y_init.shape[1]

        Y_init = Y_init.reshape(Y_init.shape[0], int(Y_init.shape[1]**0.5), int(Y_init.shape[1]**0.5))
        H, W = Y_init.shape[1], Y_init.shape[2]

        loader, _, _ = utils.create_dataloader(dataset, dev=dev, batch_size=args.batch_size)

        for n in range(n_xp):
            print(f"------ Running {n+1}th experiment ------")
            mses, sads = run_one_xp_trainers(mses, sads, i_dataset, n, Y_init, B, c, N, H, loader, dataset, args, dev)
        
        mean_mses = torch.mean(mses, dim=2)
        std_mses = torch.std(mses, dim=2)
        mean_sads = torch.mean(sads, dim=2)
        std_sads = torch.std(sads, dim=2)

        model_list = ["DeepTrans"]#, "CNN_linear", "CNNAEU", "DeepTrans"]
        trainers = ["SupervisedTrainer"]
        for i_model, model in enumerate(model_list):
            for i_trainer, trainer in enumerate(trainers):
                wandb.log({f"{dataset}_{model}_{trainer} MSE": mean_mses[i_model, i_trainer]})
                wandb.log({f"{dataset}_{model}_{trainer} SAD": mean_sads[i_model, i_trainer]})
                wandb.log({f"{dataset}_{model}_{trainer} MSE_std": std_mses[i_model, i_trainer]})
                wandb.log({f"{dataset}_{model}_{trainer} SAD_std": std_sads[i_model, i_trainer]})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--patch_size", default=5, type=int)
    parser.add_argument("--n_xp", default=5, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn')

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