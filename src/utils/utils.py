import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import scipy.io as io
import os
import wandb
from io import BytesIO
from PIL import Image
from code_christophe.munkres import Munkres

import src.utils.losses as losses

def order_endmembers(tensor_gt, tensor_hat, tensor2_hat=None):
    """
    Uses scipy linear_sum_assignement algorithm to reorder tensor_hat columns to match tensor_gt
    Tensors must be of shape (batch, D, X) or (D, X) where D is the axis along which to reorder
    tensor_2_hat is another tensor that can be reordered based on the tensor_hat reordering (for abundances)
    """
    is_batched = True
    if tensor_hat.dim() == 2:
        is_batched = False
        tensor_hat = tensor_hat.unsqueeze(0)
    if tensor_gt.dim() == 2:
        is_batched = False
        tensor_gt = tensor_gt.unsqueeze(0)

    tensor_hat_ordered = torch.zeros_like(tensor_hat)
        
    if tensor2_hat != None:
        if tensor2_hat.dim() < 4:
            tensor2_hat = tensor2_hat.unsqueeze(0)

        tensor2_hat_ordered = torch.zeros_like(tensor2_hat)

    for b in range(tensor_hat.size()[0]):

        # Normalize the tensors
        tensor_gt_norm = F.normalize(tensor_gt[b], p=2.0, dim=1)  # Normalize along reordered axis
        tensor_hat_norm = F.normalize(tensor_hat[b], p=2.0, dim=1)

        # Compute cost matrix (cosine distance)
        cost_matrix = torch.acos(torch.clamp(tensor_gt_norm @ tensor_hat_norm.T, -1.0, 1.0))
        cost_matrix_np = cost_matrix.cpu().numpy() 

        # Solve assignment problem
        _, col_ind = linear_sum_assignment(cost_matrix_np)

        # Reorder E_hat to match E_gt
        tensor_hat_ordered[b] = tensor_hat[b, col_ind]

        if tensor2_hat != None:
            tensor2_hat_ordered[b] = tensor2_hat[b, col_ind]

    if tensor2_hat != None:
        if is_batched:
            return tensor_hat_ordered, tensor2_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], tensor2_hat_ordered[0], col_ind
    else:
        if is_batched:
            return tensor_hat_ordered, col_ind
        else:
            return tensor_hat_ordered[0], col_ind
    
def order_endmembers_(E_gt, E_hat, A_hat=None):
    if E_hat.dim() == 2:
        E_hat = E_hat.unsqueeze(0)
    if E_gt.dim() == 2:
        E_gt = E_gt.unsqueeze(0)
        
    if A_hat != None:
        if A_hat.dim() < 4:
            A_hat = A_hat.unsqueeze(0)
        A_hat_corr = torch.zeros_like(A_hat)
    
    E_hat_corr_norm = torch.zeros_like(E_hat)
    E_hat_corr = torch.zeros_like(E_hat)
    indices = torch.zeros((E_hat.size()[0],E_hat.size()[2]))

    for batch in range(E_gt.size()[0]):
        E0 = F.normalize(E_gt[batch,:,:],p=2.0,dim=0)
        E = F.normalize(E_hat[batch,:,:],p=2.0,dim=0)
        E0 = E0.to(torch.float32)   
        E = E.to(torch.float32)
        
        dot_products = E0.T @ E
        costmat = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        EPerm = torch.zeros(E0.shape, dtype=torch.float32)
        EPerm_norm = torch.zeros(E0.shape)
        perm_indices = torch.zeros(E0.shape[1])
    
        if A_hat != None:
            APerm = torch.zeros(A_hat[batch].shape)
        
        for c in range(E0.shape[1]):
            EPerm[:,c] = E_hat[batch,:,Jperm[c][1]]
            EPerm_norm[:,c] = E_gt[batch, :,Jperm[c][1]]
            perm_indices[c] = Jperm[c][1]
            if A_hat != None:
                APerm[c] = A_hat[batch, Jperm[c][1]]
        
        E_hat_corr_norm[batch,:,:] = EPerm_norm
        E_hat_corr[batch,:,:] = EPerm
        indices[batch,:] = perm_indices
        if A_hat != None:
            A_hat_corr[batch] = APerm
    indices = indices.type(torch.int64)

    if A_hat != None:
        return E_hat_corr[0], A_hat_corr[0], indices
    else:
        return E_hat_corr[0], indices

def crop_patch_image(Y, patch_size, A=None):
    """
    Crops the input HSI to a square with the max amount of pixels multiple of patch_size
    
    Args:
        Y: Input HSI to be cropped
        patch_size: The patch size with which to split the image
        A: The abundance maps to be cropped
    """
    
    batch = None
    if Y.dim() != 3:
        B, N = Y.shape
    else:
        batch, B, N = Y.shape
        
    k = int((N**0.5)//patch_size)
    
    if batch != None:
        Y = oneD_to_2d(Y)
        s = k*patch_size
        Y = Y[:, :, :s, :s]
        Y = Y.reshape(batch, B, s**2)
        
        if A != None:
            A = oneD_to_2d(A)
            A = A[:, :, :s, :s]
            A = A.reshape(batch, A.shape[1], s**2)
            return Y, A
        else:
            return Y
    else:
        Y = oneD_to_2d(Y)
        s = k*patch_size
        Y = Y[:, :s, :s]
        Y = Y.reshape(B, s**2)
        
        if A != None:
            A = oneD_to_2d(A)
            A = A[:, :s, :s]
            A = A.reshape(A.shape[0], s**2)
            return Y, A
        else:
            return Y

def oneD_to_2d(Y, H=None, W=None):
    """
    Reshapes 1D tensors to 2D image
    If H and W are not passed, assumes the image is square
    
    Args:
        Y: input tensor to be reshaped
    """
    is_batched = Y.dim()==3
    if is_batched:
        batch, B, N = Y.shape
        if H != None:
            return Y.reshape(batch, B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(batch, B, H, H)
    else:
        B, N = Y.shape
        if H != None:
            return Y.reshape(B, H, W)
        
        else:
            H = int(N**0.5)
            return Y.reshape(B, H, H)

def sum_to_one(Y, is_endmember=False):
    """
    Normalizes a tensor of type Y, A, or E (batched or not) so that:
    - For Y or A: each pixel (i,j) sums to 1 across the channel dimension.
    - For E: each tensor[:, i] sums to 1 across the last dimension.
    """
    Y = Y.clone()

    if is_endmember:
        added_batch = False
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
            added_batch = True

        # Sum over last dimension for each B
        sums = Y.sum(dim=2, keepdim=True)
        Yn = Y / sums
    
    else:
        added_batch = False
        if Y.dim() == 3:
            Y = Y.unsqueeze(0)
            added_batch = True

        # Sum over channels for each pixel
        sums = Y.sum(dim=1, keepdim=True)
        Yn = Y / sums
        
    if added_batch:
        # Remove batch dimension if it was added
        Yn = Yn.squeeze(0)

    return Yn

def normalize(X, is_endmember=False, abundance_per_channel=False):
    """
    Normalizes batched tensors

    If is_endmember -> set max to 1 for every endmember
    If is Y or A (must be of shape (batch, B/c, H, W)) -> divides by frobenius norm
    """

    if is_endmember:
        X_norm = X/torch.max(X, dim=0).values

    else:
        if abundance_per_channel:
            X_norm = X/ X.amax(dim=(1,2), keepdim=True)
        else:
            X_norm = X/torch.norm(X)

    return X_norm

def load_dataset(dataset, path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/", dtype=torch.float):
    data_path = path + dataset + ".mat"
    data = io.loadmat(data_path)
        
    Y = torch.tensor(data['Y'], dtype=dtype)
    A = torch.tensor(data['A'], dtype=dtype)
    E = torch.tensor(data['E'], dtype=dtype)

    return Y, E, A

class HSI_dataset(Dataset):    
    def __init__(self, dataset, path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/", patch_size=None, dtype=None, deeptrans=False):
        
        self.dataset_name = dataset
        
        if dtype is None:
            dtype = torch.float

        data_path = path + dataset + ".mat"
        data = io.loadmat(data_path)
        
        self.B, self.c = (data['M'].shape[0], data['M'].shape[1]) if deeptrans else (data['E'].shape[0], data['E'].shape[1])
        if patch_size != None:
            self.col = patch_size
        else:
            self.col = data["Y"].shape[1]
        
        self.Y = torch.tensor(data['Y'], dtype=dtype)
        self.A = torch.tensor(data['A'], dtype=dtype)
        self.E = torch.tensor(data['M'], dtype=dtype) if deeptrans else torch.tensor(data['E'], dtype=dtype)
        
        self.B, self.N = self.Y.shape
        self.n = int(self.N ** 0.5)
        self.c = self.E.shape[1]
        
        self.patch_size = patch_size
        if patch_size != None:
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
   
def create_dataloader(dataset, path="/home/ids/edabier/HSU", dev="cpu", train_split=None, patch_size=None, batch_size=1, dtype=torch.float32, deeptrans=False):
    """
    Creates dataloader(s) for a given dataset
    
    Args:
        dataset (str): the name of the dataset to use
        dev: the device on which to pass the loaders
        train_split (float, optional): how much of the dataset to use for the training and testing sets
        patch_size (int): whether or not to patch the input HSI
    """
    path += "/SS-HSU_benchmark/datasets/"

    if patch_size is None:
        dataset = HSI_dataset(dataset, path, dtype=dtype, deeptrans=deeptrans)
    else:
        dataset = HSI_dataset(dataset, path, patch_size, dtype=dtype, deeptrans=deeptrans)
        
    if train_split != None:
        generator = torch.Generator(dev)
        train_set, test_set = random_split(dataset, lengths=[train_split, 1-train_split], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        return train_loader, test_loader, dataset.B, dataset.col
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        return train_loader, dataset.B, dataset.col
    
def save_model(model, optimizer, directory, name, epoch, is_permanent=False):
    """
    Overwrite the previous checkpoint save if not is_permanent, otherwise, saves a new version of the model
    """
    model_name = model.__class__.__name__
    if model_name == "NALMU" or model_name == "RALMU":
        model_name += str(model.T)

    if is_permanent:
        # Save a permanent copy of the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(directory, f'{model_name}_{name}_lr_{optimizer.param_groups[-1]["lr"]}_epoch_{epoch}.pt'))
        print(f'Saved permanent model {model_name}_{name}_lr_{optimizer.param_groups[-1]["lr"]}_epoch_{epoch}.pt')
    else:
        # Overwrite the temporary model save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(directory, f'{model_name}_{name}_lr_{optimizer.param_groups[-1]["lr"]}.pt'))
        # print("Saved checkpoint model")
        
def load_checkpoint(path, model, optimizer):
    """
    Loads the last training checkpoint of the model
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")
    return start_epoch
