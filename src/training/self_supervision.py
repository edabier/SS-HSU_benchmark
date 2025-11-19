import torch
import torch.nn as nn
from torchmetrics.functional.image import spectral_angle_mapper
import data_augmentation
import utils.extractor as extractor

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
            
            loss = self.criterion(y, y_hat)
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
    
    def criterion(y_gt, y_hat, r, n):
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
            
            loss = self.criterion(y, y_hat, r, n)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return a_hat, e_hat, train_losses
    

class GeneratedDataset():
    """
    Uses the input HSI to generate an extended dataset based on Hadjeres et al. 2024
    """
    
    def __init__(self, model, dataset_size, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        self.model = model
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            
        if criterion is not None:
            self.criterion = criterion
    
    def create_dataset(self, y, n_vca, n_em, n_aug, c_var=0.4):
        """
        - We first run n times the VCA algorithm to extract different EM
        - We group them and remove duplicate with a K-means algorithm to construct a library
        - We augment this library by generating variations of each material with piece-wise affine functions
        - We select the average spectra of each material of the augmented library to create the EM matrix
        - We apply FCLS to this EM matrix and the y HSI to obtain an estimation of abundance map
        - We estimate the parameters of a Dirichlet mixture to model the abundance distribution of every material
        To do this, we use an Expectation-Maximization algorithm, and we find the number of modes with the AIC 
        
        => What does this distribution represents? The pixel distribution of the materials on the image?
        
        - We create a new point in the dataset by randomly picking:
            - an EM matrix from the library E
            - an abundance map following the estimate Dirichlet mixture distribution A
            - a mixed image y obtained by: y = EA + N (adding noise to have SNR=30dB)
        
        Args:
            y: the input HSI to unmix and with which to create a dataset
            n_vca (int): the number of times to run the VCA
            n_aug (int): the number of variations of each spectra of the library to create
            c_var (float, optional): the variability coefficient (default: 0.4)
        """
        h, w, c = y.shape
        
        self.dataset = {"E": [], "A": [], "Y": []}
        
        vca = extractor.VCA()
        endmember_lib = torch.tensor([])

        # We run n_vca times the VCA extraction to get n_vca*n_em endmembers
        for _ in range(n_vca):
            e = vca.extract_endmembers(y, c=n_em)
            endmember_lib = torch.cat((endmember_lib, e), dim=1)

        # We remove duplicate ems and normalize them
        unique_spectra = vca.remove_duplicates(endmember_lib, tol=1e-4)
        norms = torch.linalg.norm(unique_spectra, dim=1, keepdim=True)
        unique_spectra_norm = unique_spectra / norms

        # We use Kmeans to cluster the ems to create exactly n_em categories
        centers, memberships = vca.group_spectra_kmeans(unique_spectra_norm.T, n_clusters=n_em)
        grouped_lib = vca.group_spectra_by_cluster(unique_spectra_norm, memberships) # shape (n_em, nb_spectra_in_cluster, b)

        # We augment the number of ems in each cluster by running n_aug times the augmentation function
        augmented_lib = [
            torch.cat([torch.stack([vca.augment_spectrum(group[:, i]) for _ in range(n_aug)], dim=1)
                    for i in range(group.shape[1])], dim=1)
        for group in grouped_lib]
        
        # We average the ems of each cluster to find an average E matrix
        e_avg = torch.stack([torch.mean(augmented_lib[i], dim=1, keepdim=True) for i in range(n_em)], dim=1).squeeze(2)
        
        for i in range(self.dataset_size):
            
            # TO DO: implement/ retrieve dataset creation code
            
            self.dataset.E.append(i)
            self.dataset.A.append(i)
            self.dataset.Y.append(i)
    
    def criterion(self, e_gt, e_hat, a_gt, a_hat):
        """
        Computes the loss between the predicted and target E and A
        """
        
        mse = nn.MSELoss()
        
        loss_e = spectral_angle_mapper(e_gt, e_hat)
        loss_a = mse(a_gt, a_hat)**0.5
        
        return loss_e + loss_a
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            
            for i in range(len(self.dataset.E)):
                
                e_gt = self.dataset.E[i]
                a_gt = self.dataset.A[i]
                y_gt = self.dataset.Y[i]
                
                self.optimizer.zero_grad()
                
                a_hat, e_hat, y_hat = self.model.unmix(y_gt)
                
                loss = self.criterion(e_gt, e_hat, a_gt, a_hat)
                train_losses.append(loss)
                
                loss.backward()
                self.optimizer.step()
            
        return a_hat, e_hat, train_losses
    

"""
/!\ TO DO: make the loss def of Two Stages and Generated Dataset act as self.criterion
"""