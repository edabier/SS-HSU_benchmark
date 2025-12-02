import torch
import torch.nn as nn
import wandb
import argparse

import src.models.models as models
import src.training.self_supervision as ssl
import src.utils.utils as utils

"""
This code runs the training of the specified model on the specified dataset using the specified training strategy

We can choose between 5 models to train:  
    - CNNAEU
    - CNN + linear decoder
    - Transformer AE
    - NALMU
    - RALMU

We can choose between 4 training strategies:
    - DIP
    - Two stages net
    - Synthetic generated dataset semi-supervised
    - Contrastive learning
"""

def main(c, patch_size, model, training_strat, split, dataset, lr, epochs, batch_size):
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"

    print(f"Start of the script, device = {dev}")
    
    train_loader, test_loader, B, col = utils.create_dataloader(dataset=dataset, dev=dev, train_split=split, patch_size=patch_size, batch_size=batch_size)
    
    if model == "CNNAEU":
        model = models.CNNAEU(B=B, c=c)
    elif model == "CNN_linear":
        model = models.CNNAE_linear(B=B, c=c, patch_size=patch_size)
    elif model == "Transformer":
        model = models.Transformer_AE(B=B, c=c, im_size=col, patch_size=patch_size)
    elif model == "NALMU":
        model = models.NALMU(b=B, c=c, N=col**2)
    else:
        model = models.RALMU(B=B, c=c, im_size=col)
    
    if training_strat == "RE":
        trainer = ssl.ReconstructionError(model, epochs=epochs, lr=lr, batch_size=batch_size)
    if training_strat == "DIP":
        trainer = ssl.DIP(model, epochs=epochs, lr=lr, batch_size=batch_size)
    elif training_strat == "TwoStagesNet":
        trainer = ssl.TwoStagesNet(model, B, epochs=epochs, lr=lr, batch_size=batch_size)
    elif train_strat == "GeneratedDataset":
        trainer = ssl.GeneratedDataset(model, epochs=epochs, lr=lr, batch_size=batch_size)
        trainer.create_dataset(dataset, c=c)
    else:
        projection_head = None # Place here the NN model chosen for the projection before the contrastive loss
        trainer = ssl.ContrastiveLearning(model, projection_head, epochs=epochs, lr=lr, batch_size=batch_size)
    
    e_hat, a_hat, train_losses = trainer.train(train_loader)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default=4, type=int)
    parser.add_argument("--patch", default=None, type=int)
    parser.add_argument("--model", default="CNNAEU", type=str)
    parser.add_argument("--train_strat", default="DIP", type=str)
    parser.add_argument("--split", default=None, type=float)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()

    c = args.c
    patch = args.patch
    model = args.model
    train_strat = args.train_strat
    split = args.split
    dataset = args.dataset
    
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    
    print(f"Starting training {model} on {dataset} using strategy {train_strat} with arguments: lr={lr}, epochs={epochs}, batch_size={batch_size}")
    
    main(c, patch_size=patch, model=model, training_strat=train_strat, split=split, dataset=dataset, lr=lr, epochs=epochs, batch_size=batch_size)