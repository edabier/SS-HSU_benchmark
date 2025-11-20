import torch
import torch.nn as nn
import wandb
import argparse
import src.models.models as models
import src.training.self_supervision as training

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

def main(c, B, model, training_strat, dataset, lr, epochs, batch_size):
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"

    print(f"Start of the script, device = {dev}")
    
    if model == "CNNAEU":
        model = models.CNNAEU(epochs=epochs, lr=lr)
    elif model == "CNN_linear":
        model = models.CNNAE_linear(epochs=epochs, lr=lr)
    elif model == "Transformer":
        model = models.Transformer_AE(c, B)
    elif model == "NALMU":
        model = models.NALMU(b=B, c=c)
    else:
        model = models.RALMU(B=B, c=c)
        
    if training_strat == "DIP":
        trainer = training.DIP(model, epochs=epochs, lr=lr, batch_size=batch_size)
    elif training_strat == "TwoStagesNet":
        trainer = training.TwoStagesNet(model, B, epochs=epochs, lr=lr, batch_size=batch_size)
    elif train_strat == "GeneratedDataset":
        trainer = training.GeneratedDataset(model, epochs=epochs, lr=lr, batch_size=batch_size)
        trainer.create_dataset(dataset, c=c)
    else:
        projection_head = None # Place here the NN model chosen for the projection before the contrastive loss
        trainer = training.ContrastiveLearning(model, projection_head, epochs=epochs, lr=lr, batch_size=batch_size)
    
    trainer.train(dataset)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default=4, type=int)
    parser.add_argument("--B", default=200, type=int)
    parser.add_argument("--model", default="CNNAEU", type=str)
    parser.add_argument("--train_strat", default="DIP", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()

    c = args.c
    B = args.B
    model = args.model
    train_strat = args.train_strat
    dataset = args.dataset
    
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    
    print(f"Starting training {model} on {dataset} using strategy {train_strat} with arguments: lr={lr}, epochs={epochs}, batch_size={batch_size}")
    
    main(c, B, model, train_strat, dataset, lr, epochs, batch_size)