import torch
import torch.nn as nn
from torchmetrics.functional.image import spectral_angle_mapper
import data_augmentation

class DIP():
    """
    Defines a Deep Image Prior-type of training based on Ulyanov et al. 2020.
    We optimize the model to reconstruct an input image y from random gaussian noise:
    
    y* = min_f || y_gt - f(z) ||

    Args:
        model: the model to train
        criterion: the function to optimize by training the model
        optimizer: the optimizer to use for the training, by default, it uses AdamW (default: None)
        epochs (int, optional): the number of training epochs to do (default: 200)
        lr (float, optional): the learning rate of the training (default: 0.001)
        batch_size (int optional): the batch size to train the model (default: 1)
    """
    def __init__(self, model, criterion, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            z = torch.randn_like(y)
            a_hat, e_hat, y_hat = self.model.unmix(z)
            
            loss = self.criterion(y_hat, y)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return a_hat, e_hat, train_losses
    

class TwoStagesNet():
    """
    Defines a Two stages Net-type of training based on Vijayashekhar et al.2022
    We optimize the model to reconstruct an input image y and force it to be a good denoiser at the same time
    We create a small MLP that is trained to denoise the output of the model:
    
    y -> model(y) = r -> r+n -> MLP(r+n) -> y_hat
    
    We train the entire model (input model + MLP)

    Args:
        model: the model to train
        criterion: the function to optimize by training the model, by default, we use the loss defined in the article (default: None)
        optimizer: the optimizer to use for the training, by default, it uses AdamW (default: None)
        epochs (int, optional): the number of training epochs to do (default: 200)
        lr (float, optional): the learning rate of the training (default: 0.001)
        batch_size (int optional): the batch size to train the model (default: 1)
    """
    def __init__(self, model, L, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            
        if criterion is not None:
            self.criterion = criterion
            
        self.denoiser = nn.Sequential(
            nn.Linear(L, 120), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(120, 90), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(90, 45), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(45, L))
    
    def loss(y_hat, y_gt, r, n):
        """
        The loss is the sum of:
        - MSE(y_hat, y_gt)
        - MSE(r+n, y_gt)
        - SAD(r+n, y_gt)
        """
        mse = nn.MSELoss()
        
        loss_forward = mse(y_gt, y_hat)
        loss_denoiser = mse(y_gt, (r+n))
        loss_sad = spectral_angle_mapper(y_gt, (r+n))
        
        return loss_forward + loss_denoiser + loss_sad
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            a_hat, e_hat, r = self.model.unmix(y)
            n = torch.randn_like(y)
            r += n
            y_hat = self.denoiser(r)
            
            loss = self.criterion(y_hat, y, r, n)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return a_hat, e_hat, train_losses
    

class GeneratedDataset():
    
    def __init__(self):
        pass
    
    def create_dataset(y, n_iter_vca):
        
        pass
    
    def train():
        pass