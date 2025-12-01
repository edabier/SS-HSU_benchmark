import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import math
import scipy.io as io

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, targets, predictions):
        targets_norm = normalize(targets,p=2.0,dim=1)
        predictions_norm = normalize(predictions,p=2.0,dim=1)
        matConfusion = torch.bmm(torch.transpose(targets_norm, 1, 2),predictions_norm)
        
        diagBatch = torch.diagonal(matConfusion,dim1=1,dim2=2) # Prend la diagonale pour chaque mini-batch
        
        return -torch.sum(diagBatch)/(targets.size()[0]*targets.size()[2]) # Independant de la taille des mini-batchs et du nombre de sources
 
class toutesLoss(nn.Module):
    # For abundance matrices. To use it on EM, transpose the two inputs. (Doesn't correct permutations)
    """
    Args:
        optLoss (int, optional): 
            - 0 MSE on E and A
            - 1 SAD on E and A
            - 2 MSE on normalized E and A
            - 3 SAD on E and MSE on A
            - 4 SAD on E and NMSE on A
            - 5 SAD on E term-wise NMSE on A
            - 6 NMSE on A

    """
    def __init__(self,optLoss=0):
        super(toutesLoss, self).__init__()
        self.optLoss = optLoss
        
        if optLoss==1:
            self.criterion = SADLoss()
        elif optLoss==0 or optLoss==2:
            self.criterion = nn.MSELoss(reduction='mean')
            
        elif optLoss==3:
            self.critSAD = SADLoss()
            self.critMSE = nn.MSELoss(reduction='mean')
            
        elif optLoss==4 or optLoss== 5 or optLoss==6:
            self.critSAD = SADLoss()
            self.critMSE = nn.MSELoss(reduction='sum')
        
    def forward(self,E,E_pred,A,A_pred):
        if self.optLoss==1:
            train_E = self.criterion(E,E_pred)
            train_A = self.criterion(torch.transpose(A, 1, 2),torch.transpose(A_pred, 1, 2))
            
            train_loss = train_E + train_A
            
        elif self.optLoss==0:
            train_E = self.criterion(E,E_pred)
            train_A = self.criterion(A,A_pred)
            
            train_loss = train_E + train_A
            
        elif self.optLoss==2:
            E_norm = normalize(E,p=2.0,dim=1)
            E_pred_norm = normalize(E_pred,p=2.0,dim=1)
            A_norm = normalize(A,p=2.0,dim=2)
            A_pred_norm = normalize(A_pred,p=2.0,dim=2)

            train_E = self.criterion(E_norm,E_pred_norm)
            train_A = self.criterion(A_norm,A_pred_norm)
            
            train_loss = train_E + train_A

        elif self.optLoss==3:
            train_E = self.critSAD(E,E_pred)
            train_A = 1000*self.critMSE(A,A_pred)
            
            train_loss = train_E + train_A
            
        elif self.optLoss==4:
            train_E = self.critSAD(E,E_pred)
            train_A = self.critMSE(A,A_pred)/(torch.norm(A)**2)
            
            train_loss = train_E + train_A
            
        elif self.optLoss==5:
            train_E = self.critSAD(E,E_pred)
            train_A = 0
            
            for ii in range(A.size()[1]):
                train_A += self.critMSE(A[:,ii,:],A_pred[:,ii,:])/(torch.norm(A[:,ii,:])**2)
            
            train_loss = train_E + train_A
            
        elif self.optLoss==6:# Pas de loss sur E, seulement sur A
            train_A = self.critMSE(A,A_pred)/(torch.norm(A)**2)
            train_E = torch.zeros(1)
            
            train_loss = train_A
            
        return train_loss,train_E,train_A
    
class HSI_dataset(Dataset):    
    def __init__(self, dataset, patch_size=None, dtype=None):
        
        if dtype is None:
            dtype = torch.float32

        data_path = "datasets/" + dataset + ".mat"
        data = io.loadmat(data_path)
        
        if dataset == 'samson':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 95
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'jasper':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 100
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'urban':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 307
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'apex':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 110
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_1':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 75
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_2':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 100
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_3':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1] #H = 105
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        
        self.Y = torch.tensor(data['Y'], dtype=dtype)
        self.A = torch.tensor(data['A'], dtype=dtype)
        self.E = torch.tensor(data['E'], dtype=dtype)
        
        self.B, self.N = self.Y.shape
        self.n = int(self.N ** 0.5)
        self.c = self.E.shape[1]
        
        self.patch_size = patch_size
        if patch_size is not None:
            # number of patches in one row/column (after padding)
            self.num_rows = math.ceil(self.n / patch_size)
            self.num_cols = self.num_rows  # square image assumption
            # update dataset length to total patches:
            self.total_patches = self.num_rows * self.num_cols
        else:
            # If no patches, dataset has one item (the whole image)
            self.total_patches = 1

    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        if self.patch_size is None:
            return self.Y, self.E, self.A
        
        # If patching, ensure idx is in range:
        if idx < 0 or idx >= self.total_patches:
            raise IndexError(f"Index {idx} out of range")
        
        # Compute patch row/col (row-major ordering)
        row = idx // self.num_cols
        col = idx % self.num_cols

        # Reflect-pad Y and A to have size divisible by patch_size
        pad_h = (self.patch_size - (self.n % self.patch_size)) % self.patch_size
        pad_w = pad_h  # square images
        # Reshape to 2D images for padding
        Y2d = self.Y.view(self.B, 1, self.n, self.n)      # shape (B,1,n,n)
        A2d = self.A.view(1, self.c, self.n, self.n)      # shape (1,c,n,n)
        # Apply reflection padding on bottom/right only
        # pad = (left, right, top, bottom)
        pad = (0, pad_w, 0, pad_h)
        Y_pad = F.pad(Y2d, pad, mode='reflect')  # shape (B,1,n+pad_h,n+pad_w)
        A_pad = F.pad(A2d, pad, mode='reflect')  # shape (1,c,n+pad_h,n+pad_w)

        # Calculate starting coordinates of this patch
        y_start = row * self.patch_size
        x_start = col * self.patch_size

        # Slice out the patch (size patch_size x patch_size)
        Y_patch = Y_pad[:, :, y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size]  # shape (B,1,p,p)
        A_patch = A_pad[:, :, y_start : y_start + self.patch_size,
                            x_start : x_start + self.patch_size]  # shape (1,c,p,p)
        
        # # Remove the singleton dimensions and flatten spatial dims
        Y_patch = Y_patch.reshape(1, self.B, self.patch_size**2).squeeze(0) # shape (1, B, p*p)
        A_patch = A_patch.reshape(1, self.c, self.patch_size**2).squeeze(0) # shape (1, c, p*p)

        # Return the patch and the full E
        return Y_patch, self.E, A_patch
   
def create_dataloader(dataset, dev, train_split=None, patch_size=None, batch_size=1):
    """
    Creates dataloader(s) for a given dataset
    
    Args:
        dataset (str): the name of the dataset to use
        dev: the device on which to pass the loaders
        train_split (float, optional): how much of the dataset to use for the training and testing sets
        patch_size (int): whether or not to patch the input HSI
    """
    if patch_size is None:
        
        dataset = HSI_dataset(dataset)
        train_loader = DataLoader(dataset, batch_size)
        return train_loader
    
    else:
        dataset = HSI_dataset(dataset, patch_size)
        
        generator = torch.Generator(dev)
        train_set, test_set = random_split(dataset, lengths=[train_split, 1-train_split], generator=generator)
    
        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size)
        
        return train_loader, test_loader
    
def compute_metrics(E, A, E_hat, A_hat, rmse=False):
    
    if rmse:
        re = torch.mean(torch.sqrt(torch.mean((A - A_hat) ** 2, dim=0)))
    else:
        re = torch.mean(torch.sum((A - A_hat) ** 2, dim=1))
        
    E_norm = E / torch.norm(E, dim=0, keepdim=True)
    E_hat_norm = E_hat / torch.norm(E_hat, dim=0, keepdim=True)
    sad = torch.mean(torch.acos(torch.clamp(torch.sum(E_norm * E_hat_norm, dim=0), -1.0, 1.0)))
    
    return re, sad