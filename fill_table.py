import torch
import scipy.io as io
import argparse
import matplotlib.pyplot as plt
import wandb

import src.utils.utils as utils
import src.models.models as models
import src.training.self_supervision as ssl

def main(args):
    patch_size = args.patch_size
    n_xp = args.n_xp

    datasets = ["urban", "apex", "jasper", "samson"]

    # shape (n_datasets, n_models, n_trainers, n_xp)
    mses = torch.zeros(len(datasets), 4, 3, n_xp)
    sads = torch.zeros(len(datasets), 4, 3, n_xp)

    for i_dataset, dataset in enumerate(datasets):

        print(f"####### {dataset} #######")

        data = io.loadmat(f"/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/{dataset}.mat")
        Y_init = torch.tensor(data["Y"])
        Y_init = Y_init.to(torch.float32)
        E = torch.tensor(data["E"])
        B, c, N = E.shape[0], E.shape[1], Y_init.shape[1]

        Y_init = Y_init.reshape(Y_init.shape[0], int(Y_init.shape[1]**0.5), int(Y_init.shape[1]**0.5))
        H, W = Y_init.shape[1], Y_init.shape[2]

        loader, _, _ = utils.create_dataloader(dataset, dev, batch_size=args.batch_size)

        for n in range(n_xp):

            print(f"------ Running {n+1}th experiment ------")

            """
            Instanciating models
            """
            model_list = []

            # # MLP
            # mlpae = models.MLP_AE(B=B, c=c)
            # model_list.append(mlpae)

            # # CNN + linear decoder
            # cnnlinear = models.CNNAE_linear(B=B, c=c, patch_size=patch_size)
            # model_list.append(cnnlinear)

            # # CNNAEU
            # Y_loader, e_gt, a_gt = next(iter(loader))[0][0], next(iter(loader))[1][0], next(iter(loader))[2][0]
            # cnnaeu = models.CNNAEU(B=B, c =c)
            # cnnaeu = models.init_decoder_weights(cnnaeu, Y_loader, c, 11)
            # model_list.append(cnnaeu)
            
            # # Deep Trans
            # Y_trans = Y_init[:,:(Y_init.shape[1]//patch_size)*patch_size,:(Y_init.shape[1]//patch_size)*patch_size]
            # deep_trans = models.Transformer_AE(B=B, c=c, im_size=Y_trans.shape[1], dim=200)
            # deep_trans = models.init_decoder_weights(deep_trans, Y_loader, c)
            # model_list.append(deep_trans)

            # NALMU
            nalmu = models.NALMU(B=B, c=c, N=N)
            model_list.append(nalmu)

            # RALMU
            ralmu = models.RALMU(B=B, c=c, im_size=H)
            model_list.append(ralmu)

            """
            Instanciating trainers
            """
            for i_model, model in enumerate(model_list):

                print(f"Training {model.__class__.__name__}")

                trainers = []

                # Supervised training
                supervised_trainer = ssl.SupervisedTrainer(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
                trainers.append([supervised_trainer])

                # Reconstruction Error
                re_trainer = ssl.ReconstructionError(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
                re_trainer_trans = ssl.ReconstructionError(model, patch_size=patch_size, epochs=args.epochs, lr=args.lr)
                trainers.append([re_trainer, re_trainer_trans])

                # DIP
                dip_trainer = ssl.DIP(model, has_decoder=False, epochs=args.epochs, lr=args.lr)
                dip_trainer_trans = ssl.DIP(model, patch_size=patch_size, epochs=args.epochs, lr=args.lr)
                trainers.append([dip_trainer, dip_trainer_trans])

                # Two stages
                twos_trainer = ssl.TwoStagesNet(model, B, has_decoder=False, epochs=args.epochs, lr=args.lr)
                twos_trainer_trans = ssl.TwoStagesNet(model, B, patch_size=patch_size, epochs=args.epochs, lr=args.lr)
                trainers.append([twos_trainer, twos_trainer_trans])

                for i_trainer, trainer_ in enumerate(trainers):
                    if model.__class__.__name__ == "Transformer_AE":
                        trainer = trainer_[1]
                    else:
                        trainer = trainer_[0]

                    print(f"Starting {trainer.__class__.__name__} startegy")

                    e_hat, a_hat, train_losses = trainer.train(loader)

                    plt.plot(train_losses)
                    test = {f"{dataset}_{model.__class__.__name__}_{trainer.__class__.__name__}_train_loss": wandb.Image(plt)}
                    wandb.log(test)
                    plt.close()

                    model.load_state_dict(torch.load(f"/home/ids/edabier/HSU/SS-HSU_benchmark/models/{model.__class__.__name__}_{trainer.__class__.__name__}_{dataset}_lr_{args.lr}.pt")["model_state_dict"], strict=False)
                    
                    Y, e_gt, a_gt = next(iter(loader))[0][0], next(iter(loader))[1][0], next(iter(loader))[2][0]

                    if model.__class__.__name__ == "Transformer_AE":

                        k = int((Y.shape[1]**0.5)//patch_size)
                        Y = Y.reshape(Y.shape[0], int(Y.shape[1]**0.5), int(Y.shape[1]**0.5))
                        s = k*patch_size
                        Y = Y[:, :s, :s]
                        Y = Y.reshape(Y.shape[0], s**2)
                        a_gt = a_gt.reshape(a_gt.shape[0], int(a_gt.shape[1]**0.5), int(a_gt.shape[1]**0.5))
                        a_gt = a_gt[:, :s, :s]
                        a_gt = a_gt.reshape(a_gt.shape[0], s**2)

                    
                    A_init_disp = next(iter(loader))[2].to(torch.float32)
                    E_init_disp = torch.ones(next(iter(loader))[1].size(),dtype=torch.float32)
                    A_init = torch.ones_like(A_init_disp)
                    E_init = torch.ones_like(E_init_disp)

                    e_hat, a_hat, y_hat = model.forward(Y, E_init=E_init, A_init=A_init)

                    if e_hat.dim() == 3:
                        e_hat = e_hat.squeeze(0)
                    
                    a_hat = a_hat.reshape(a_hat.shape[1], int(a_hat.shape[2]**0.5), int(a_hat.shape[2]**0.5))
                    a_gt = a_gt.reshape(a_gt.shape[0], int(a_gt.shape[1]**0.5), int(a_gt.shape[1]**0.5))    
                    
                    try:
                        mse, sad = utils.compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=f"{model.__class__.__name__}_{trainer.__class__.__name__}_{dataset}", use_wandb=True)
                        mses[i_dataset, i_model, i_trainer, n] = torch.tensor(mse[-1]).item()
                        sads[i_dataset, i_model, i_trainer, n] = torch.tensor(sad[-1]).item()
                    except:
                        continue

    mean_mses = torch.mean(mses, dim=3)
    std_mses = torch.std(mses, dim=3)
    mean_sads = torch.mean(sads, dim=3)
    std_sads = torch.std(sads, dim=3)

    for i_dataset, dataset in enumerate(datasets):
        for i_model, model in enumerate(model_list):
            for i_trainer, trainer in enumerate(trainers):
                wandb.log({f"{dataset}_{model.__class__.__name__}_{trainer[0].__class__.__name__} MSE": mean_mses[i_dataset, i_model, i_trainer]})
                wandb.log({f"{dataset}_{model.__class__.__name__}_{trainer[0].__class__.__name__} SAD": mean_sads[i_dataset, i_model, i_trainer]})
                wandb.log({f"{dataset}_{model.__class__.__name__}_{trainer[0].__class__.__name__} MSE_std": std_mses[i_dataset, i_model, i_trainer]})
                wandb.log({f"{dataset}_{model.__class__.__name__}_{trainer[0].__class__.__name__} SAD_std": std_sads[i_dataset, i_model, i_trainer]})

if __name__ == "__main__":

    run = wandb.init(project=f"Fill_table")

    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        print(f"{torch.cuda.is_available()}")
        dev = "cpu"
    
    print(f"Starting project on dev: {dev}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--patch_size", default=5, type=int)
    parser.add_argument("--n_xp", default=5, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()
    
    main(args)