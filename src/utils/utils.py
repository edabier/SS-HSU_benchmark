import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as io
import os
import wandb
from io import BytesIO
from PIL import Image
from code_christophe.munkres import Munkres

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = F.normalize(y_true, dim=0, p=2)
            y_pred = F.normalize(y_pred, dim=0, p=2)
        else:
            y_true = F.normalize(y_true, dim=1, p=2)
            y_pred = F.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
            
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss

def numpy_alter_MSE(y_true, y_pred):
    num_em = y_true.shape[0]
    y_true = np.reshape(y_true , [num_em, -1])
    y_pred = np.reshape(y_pred , [num_em, -1])

    R = y_pred - y_true
    r = R*R
    mse = np.mean(r, axis=1)
    Average_mse = np.sum(mse) / len(mse)
    mse = np.insert(mse, num_em, Average_mse, axis=0)
    return mse

def alter_MSE(y_true, y_pred):
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    mse = torch.mean((y_pred - y_true) ** 2, dim=1)
    Average_mse = torch.mean(mse)
    mse_with_avg = torch.cat([mse, Average_mse.unsqueeze(0)])

    return mse_with_avg
   
def order_endmembers(E_gt, E_hat, A_hat=None):
    if E_hat.dim() == 2:
        E_hat = E_hat.unsqueeze(0)
    if E_gt.dim() == 2:
        E_gt = E_gt.unsqueeze(0)
        
    if A_hat is not None:
        if A_hat.dim() < 4:
            A_hat = A_hat.unsqueeze(0)
        A_hat_corr = torch.zeros_like(A_hat)
    
    E_hat_corr_norm = torch.zeros_like(E_hat)
    E_hat_corr = torch.zeros_like(E_hat)
    indices = torch.zeros((E_hat.size()[0],E_hat.size()[2])) # Premier indice : mini-batch, deuxieme : nombre de sources

    for batch in range(E_gt.size()[0]):
        E0 = F.normalize(E_gt[batch,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        E = F.normalize(E_hat[batch,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        E0 = E0.to(torch.float64)   
        E = E.to(torch.float64)
        
        dot_products = E0.T @ E
        costmat = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        EPerm = torch.zeros(E0.shape, dtype=torch.float64)
        EPerm_norm = torch.zeros(E0.shape)
        perm_indices = torch.zeros(E0.shape[1])
    
        if A_hat is not None:
            APerm = torch.zeros(A_hat[batch].shape)
        
        for c in range(E0.shape[1]):
            EPerm[:,c] = E_hat[batch,:,Jperm[c][1]]
            EPerm_norm[:,c] = E_gt[batch, :,Jperm[c][1]]
            perm_indices[c] = Jperm[c][1]
            if A_hat is not None:
                APerm[c] = A_hat[batch, Jperm[c][1]]
        
        E_hat_corr_norm[batch,:,:] = EPerm_norm
        E_hat_corr[batch,:,:] = EPerm
        indices[batch,:] = perm_indices
        if A_hat is not None:
            A_hat_corr[batch] = APerm
    indices = indices.type(torch.int64)

    if A_hat is not None:
        return E_hat_corr[0], E_hat_corr_norm[0], A_hat_corr[0], indices
    else:
        return E_hat_corr[0], E_hat_corr_norm[0], indices
 
def order_abundances(A_gt, A_hat, E_hat=None):
    if A_hat.dim() == 3:
        A_hat = A_hat.unsqueeze(0)
    if A_gt.dim() == 3:
        A_gt = A_gt.unsqueeze(0)
        
    if E_hat is not None:
        if E_hat.dim() < 3:
            E_hat = E_hat.unsqueeze(0)
        E_hat_corr = torch.zeros_like(E_hat)
    
    B, c, H, W = A_hat.shape
    A_hat = A_hat.reshape(B, c, -1).to(torch.float)
    A_gt = A_gt.reshape(B, c, -1).to(torch.float)

    A_hat_corr_gt = torch.zeros_like(A_hat)
    A_hat_corr = torch.zeros_like(A_hat)
    indices = torch.zeros((A_hat.size()[0],A_hat.size()[1]))

    for batch in range(A_gt.size()[0]):
        A_true = F.normalize(A_gt[batch], p=2.0, dim=1)  # (c, H*W)
        A_pred = F.normalize(A_hat[batch], p=2.0, dim=1)  # (c, H*W)

        cos_sim = A_true @ A_pred.T
        costmat = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))

        dot_products = A_true @ A_pred.T
        costmat = torch.acos(torch.clamp(dot_products, -1.0, 1.0))
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        APerm = torch.zeros(A_true.shape, dtype=torch.float)
        APerm_gt = torch.zeros(A_true.shape, dtype=torch.float)
        perm_indices = torch.zeros(A_true.shape[0])
    
        if E_hat is not None:
            EPerm = torch.zeros(E_hat[batch].shape)
        
        for c in range(A_true.shape[0]):
            APerm[c, :] = A_hat[batch, Jperm[c][1]]
            APerm_gt[c, :] = A_gt[batch, Jperm[c][1]]
            perm_indices[c] = Jperm[c][1]

            if E_hat is not None:
                EPerm[:, c] = E_hat[batch, :, Jperm[c][1]]

        A_hat_corr_gt[batch] = APerm_gt
        A_hat_corr[batch] = APerm
        indices[batch] = perm_indices
        if E_hat is not None:
            E_hat_corr[batch] = EPerm
    indices = indices.type(torch.int64)

    A_ordered = A_hat_corr[0]
    A_ordered_gt = A_hat_corr_gt[0]
    A_ordered = oneD_to_2d(A_ordered)
    A_ordered_gt = oneD_to_2d(A_ordered_gt)

    if A_hat is not None:
        E_ordered = E_hat_corr[0]
        return A_ordered, A_ordered_gt, E_ordered, indices
    else:
        return A_ordered, A_ordered_gt,indices

def compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=None, use_wandb=False):
    """
    Computes the SAD of predicted E and MSE of predicted A
    """
    sad = SADLoss()
    mse = nn.MSELoss(reduction = "sum")

    num_E = e_hat.shape[-1]
    n = num_E // 2
    if num_E % 2 != 0: n = n + 1

    E_ordered, _, a_hat, indices = order_endmembers(e_hat, e_gt, a_hat)
    
    sads = []
    mses = []

    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()

    for i in range(num_E):
        sad_ = sad(e_gt[0, :, i], E_ordered[:, i])
        mse_ = mse(a_gt, a_hat)/(torch.norm(a_gt)**2)
        sads.append(sad_.item())
        mses.append(mse_.item())
    sads = torch.tensor(sads)
    mses = torch.tensor(mses)
    Average_SAD = torch.mean(sads)
    Average_MSE = torch.mean(mses)
    title = f"{name} aSAD score for all E: " + format(Average_SAD, '.5f')
    st = plt.suptitle(title)

    for i in range(num_E):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(e_gt[:, i].detach().cpu(), 'r', linewidth=1.0, label='GT')
        plt.plot(E_ordered[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
        plt.legend()
        ax.set_title("SAD: " + format(sads[i], '.5f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    # st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        wandb.log({"Endmember extraction": wandb.Image(img_array)})

    fig, axes = plt.subplots(a_hat.shape[0], 2, figsize=(5, 10))
    axes[0, 0].set_title(f"{name}_pred", fontsize=12)
    axes[0, 1].set_title(f"{name}_GT", fontsize=12)

    # Plot tensor images
    for i in range(a_hat.shape[0]):
        axes[i, 0].imshow(a_hat[i].detach().cpu())
        axes[i, 0].axis('off')

        axes[i, 1].imshow(a_gt[i].detach().cpu())
        axes[i, 1].axis('off')

    # Adjust layout to reduce white space
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    plt.tight_layout()
    plt.show()

    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        wandb.log({"Abundance extraction": wandb.Image(img_array)})

    return Average_MSE, Average_SAD

def compute_metrics(E_gt, A_gt, E_hat, A_hat, Y_gt=None, Y_hat=None):
    
    sad = SADLoss()
    mse = nn.MSELoss(reduction='sum')
    
    metric_A = mse(A_gt, A_hat)/(torch.norm(A_gt)**2)
    metric_E = sad(E_gt, E_hat)

    if Y_gt is not None:
        metric_re = mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)
    
        return metric_A, metric_E, metric_re
    else:
        return metric_A, metric_E

def plot_results(E_hat, A_hat, A_gt=None, E_gt=None, model_name=None, normalize_E=False, normalize_A=False, plot_only_E=False, plot_only_A=False):
    """
    Displays the predicted endmembers and abundances
    """
    if E_hat.dim() == 3:
        print("Not taking batched E_hat")
        E_hat = E_hat[0]
    if E_gt is not None and E_gt.dim() == 3:
        print("Not taking batched E_gt")
        E_gt = E_gt[0]
    if A_hat.dim() == 4:
        print("Not taking batched A_hat")
        A_hat = A_hat[0]
    if A_gt.dim() == 4:
        print("Not taking batched A_gt")
        A_gt = A_gt[0]

    c = E_hat.shape[1]
    n_graph = c // 2
    if c % 2 != 0: n_graph = n_graph + 1

    if not plot_only_A:
        if E_gt is not None:

            E_hat, _, A_hat, _ = order_endmembers(E_gt, E_hat, A_hat)
            
            if normalize_E:
                E_hat = normalize(E_hat, dim=1)
                E_gt = normalize(E_gt, dim=1)

            fig, axes = plt.subplots(2, n_graph)
            axes = axes.flatten()
            if model_name is not None:
                plt.suptitle(f'{model_name} Endmember estimation')
            else:
                plt.suptitle('Endmember estimation')
            for i in range(c):
                ax = axes[i]
                ax.plot(E_gt[:, i].detach().cpu(), 'r', linewidth=1.0, label='GT')
                ax.plot(E_hat[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
                if i == 0:
                    plt.legend() 
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
            
            sad = SADLoss()
            print(f"SAD(E, E_hat) = {format(sad(E_gt, E_hat), '.3f')}")
        else:
            if model_name is not None:
                plt.suptitle(f'{model_name} Endmember estimation')
            else:
                plt.suptitle('Endmember estimation')
            for i in range(c):
                ax = plt.subplot(2, n_graph, i + 1)
                plt.plot(E_hat[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')

    if not plot_only_E:
        if A_gt is not None:

            if normalize_A:
                A_hat = normalize(A_hat)

            fig, axes = plt.subplots(2, A_hat.shape[0], figsize=(10, 5))
            if model_name is not None:
                plt.suptitle(f'{model_name} abundance estimation')
            else:
                plt.suptitle('Abundance estimation')
            axes[0, 0].set_title(f"Prediction", fontsize=12)
            axes[1, 0].set_title(f"GT", fontsize=12)
            for i in range(A_hat.shape[0]):
                pred = axes[0, i].imshow(A_hat[i].detach().cpu())
                axes[0, i].axis('off')

                gt = axes[1, i].imshow(A_gt[i].detach().cpu())
                axes[1, i].axis('off')
            bar = plt.colorbar(gt)
            bar = plt.colorbar(pred)

            mse = nn.MSELoss(reduction="sum")
            print(f"MSE(A, A_hat) = {format(mse(A_gt, A_hat)/torch.norm(A_gt)**2, '.3f')}")
        else:
            fig, axes = plt.subplots(1, A_hat.shape[0], figsize=(10, 5))
            if model_name is not None:
                plt.suptitle(f'{model_name} abundance estimation')
            else:
                plt.suptitle('Abundance estimation')
            axes[0].set_title(f"Prediction", fontsize=12)
            for i in range(A_hat.shape[0]):
                pred = axes[i].imshow(A_hat[i].detach().cpu())
                axes[i].axis('off')
            bar = plt.colorbar(pred)

def compare_hsis(Y_gt, Y_hat, title=None):
    """
    Displays the first 4 channels of both reconstructed and groundtruth HSIs
    Must be of shape (batch, B, H, W) or (B, H, W)
    """
    if Y_gt.dim() > 3:
        Y_gt = Y_gt.squeeze(0)
    if Y_hat.dim() > 3:
        Y_hat = Y_hat.squeeze(0)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    B, H, W = Y_gt.shape
    axes[0, 0].set_title(f"Prediction", fontsize=12)
    axes[1, 0].set_title(f"GT", fontsize=12)
    for i in range(4):
        i_th = int((i/4)*B)
        pred = axes[0, i].imshow(Y_hat[i_th].detach().cpu())
        axes[0, i].axis('off')

        gt = axes[1, i].imshow(Y_gt[i_th].detach().cpu())
        axes[1, i].axis('off')
    if title is not None:
        plt.suptitle(title)
    bar = plt.colorbar(gt)
    bar = plt.colorbar(pred)

def test_model(model, test_loader, wandb=False):
    """
    Tests the input model on the input test dataset
    """
    
    if wandb:
        run = wandb.init(
            project=f"{model.__class__.__name__}_test",
            config={
                "dataset": test_loader # Check if the dataloader can access the dataset variables to get the name
            },
        )
        
    test_metrics = {"re": [], "sad": []}
    for Y, E, A in test_loader:
        
        e_hat, a_hat, y_hat = model(Y)
            
        re, sad = compute_metrics(E, A, e_hat, a_hat)
        test_metrics["re"].append(re)
        test_metrics["sad"].append(sad)
        
        if wandb:
            wandb.log({"re": re})
            wandb.log({"sad": sad})
    
    mean_re = torch.mean(torch.tensor(test_metrics["re"]))
    mean_sad = torch.mean(torch.tensor(test_metrics["sad"]))
    
    return mean_re, mean_sad

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
    
    if batch is not None:
        Y = oneD_to_2d(Y)
        s = k*patch_size
        Y = Y[:, :, :s, :s]
        Y = Y.reshape(batch, B, s**2)
        
        if A is not None:
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
        
        if A is not None:
            A = oneD_to_2d(A)
            A = A[:, :s, :s]
            A = A.reshape(A.shape[0], s**2)
            return Y, A
        else:
            return Y

def oneD_to_2d(Y):
    """
    Reshapes 1D tensors to 2D image
    
    Args:
        Y: input tensor to be reshaped
    """
    is_batched = Y.dim()==3
    if is_batched:
        batch, B, N = Y.shape
        H = int(N**0.5)
        return Y.reshape(batch, B, H, H)
    else:
        B, N = Y.shape
        H = int(N**0.5)
        return Y.reshape(B, H, H)

def normalize(Y, dim=0):
    """
    Normalizes the input tensor along the given dimension
    For abundances, must be of shape (c, H, W)
    For endmembers, must be of shape (B, c)
    """
    # shape = Y.shape

    # In the case of abundances/ HSIs
    if Y.dim() == 3:
        max_values = torch.amax(torch.abs(Y), dim=[d for d in range(Y.dim()) if d != dim], keepdim=True)
        max_values[max_values == 0] = 1
        Y_normalized = Y / max_values
        # c, H, W = Y.shape
        # Y_reshaped = Y.reshape(c, H*W)
        # max_vals = Y_reshaped.max(dim=1)[0].view(c, 1, 1)
        # eps = 1e-8
        # Y_normalized = Y / (max_vals + eps)
        # Y_normalized = torch.clamp(Y_normalized, 0, 1)

    elif Y.dim() == 2:
        max_values = torch.amax(torch.abs(Y), dim=[d for d in range(Y.dim()) if d != dim], keepdim=True)
        max_values[max_values == 0] = 1
        Y_normalized = Y / max_values

    return Y_normalized

class HSI_dataset(Dataset):    
    def __init__(self, dataset, path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/", patch_size=None, dtype=None):
        
        self.dataset_name = dataset
        
        if dtype is None:
            dtype = torch.float32

        data_path = path + dataset + ".mat"
        data = io.loadmat(data_path)
        
        if dataset == 'samson':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'jasper':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'urban':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'apex':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_1':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_2':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
            if patch_size is not None:
                self.col = patch_size
            else:
                self.col = data["Y"].shape[1]
        elif dataset == 'simulee_3':
            self.B, self.c = data['E'].shape[0], data['E'].shape[1]
            
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
   
def create_dataloader(dataset, path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/", dev="cpu", train_split=None, patch_size=None, batch_size=1, dtype=torch.float32):
    """
    Creates dataloader(s) for a given dataset
    
    Args:
        dataset (str): the name of the dataset to use
        dev: the device on which to pass the loaders
        train_split (float, optional): how much of the dataset to use for the training and testing sets
        patch_size (int): whether or not to patch the input HSI
    """
    if patch_size is None:
        dataset = HSI_dataset(dataset, path, dtype=dtype)
    else:
        dataset = HSI_dataset(dataset, path, patch_size, dtype=dtype)
        
    if train_split is not None:
        generator = torch.Generator(dev)
        train_set, test_set = random_split(dataset, lengths=[train_split, 1-train_split], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)
        return train_loader, test_loader, dataset.B, dataset.col
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size)
        return train_loader, dataset.B, dataset.col

class HyperspectralDataset(Dataset):
    def __init__(self, dataset_name, data_path, patch_size):
        # Load the .mat file
        data = io.loadmat(data_path + dataset_name + ".mat")
        Y = data["Y"] # (B, N)
        E = data["E"] # (B, c)
        A = data["A"] # (c, N)

        # Mirror the image and abundance maps
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.E = torch.tensor(E, dtype=torch.float32)
        self.B = self.Y.shape[0]
        self.N = self.Y.shape[1]
        self.c = self.A.shape[0]
        self.patch_size = patch_size
        self.patch_N = patch_size*patch_size

        self.Y_2d = oneD_to_2d(self.Y) # (B, H, W)
        self.A_2d = oneD_to_2d(self.A) # (c, H, W)
        
        self.mirror_Y = self.mirror_2d(self.Y_2d, patch_size)
        self.mirror_A = self.mirror_2d(self.A_2d, patch_size)

        # Generate all pixel positions
        H, W = self.Y_2d.shape[1], self.Y_2d.shape[2]
        max_x = self.mirror_Y.shape[1] - patch_size
        max_y = self.mirror_Y.shape[2] - patch_size

        # Generate all pixel positions within the valid range
        rows = torch.arange(patch_size // 2, max_x)
        cols = torch.arange(patch_size // 2, max_y)
        self.all_positions = torch.cartesian_prod(rows, cols)
    
    def mirror_2d(self, tensor_2d, patch_size):
        # tensor_2d: (B, H, W) or (c, H, W)
        padding = patch_size #// 2
        H, W = tensor_2d.shape[1], tensor_2d.shape[2]

        # Create a zero-padded tensor with extended padding
        mirror = torch.zeros(
            (tensor_2d.shape[0], H + 2 * padding, W + 2 * padding),
            dtype=tensor_2d.dtype, device=tensor_2d.device
        )

        # Central region
        mirror[:, padding:(padding + H), padding:(padding + W)] = tensor_2d

        # Left mirroring
        for i in range(padding):
            mirror[:, padding:(H + padding), i] = tensor_2d[:, :, padding - i - 1]

        # Right mirroring
        for i in range(padding):
            mirror[:, padding:(H + padding), W + padding + i] = tensor_2d[:, :, W - 1 - i]

        # Top mirroring
        for i in range(padding):
            mirror[:, i, :] = mirror[:, padding * 2 - i - 1, :]

        # Bottom mirroring
        for i in range(padding):
            mirror[:, H + padding + i, :] = mirror[:, H + padding - 1 - i, :]

        return mirror
    
    def __len__(self):
        return len(self.all_positions)
    
    def __getitem__(self, idx):
        x, y = self.all_positions[idx]

        # Extract Y patch: (B, patch_size, patch_size) -> (B, patch_N)
        Y_patch = self.mirror_Y[:, x:(x + self.patch_size), y:(y + self.patch_size)]
        Y_patch = Y_patch.reshape(self.B, self.patch_N)

        # Extract A patch: (c, patch_size, patch_size) -> (c, patch_N)
        A_patch = self.mirror_A[:, x:(x + self.patch_size), y:(y + self.patch_size)]
        A_patch = A_patch.reshape(self.c, self.patch_N)

        # E is the same for all patches: (B, c)
        E_patch = self.E

        return (
            Y_patch,
            E_patch,
            A_patch
        )

def get_dataloader(dataset_name, patch_size, batch_size, n_workers=0, data_path="/home/ids/edabier/HSU/SS-HSU_benchmark/datasets/"):
    dataset = HyperspectralDataset(dataset_name, data_path, patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                         num_workers=n_workers, pin_memory=True)
    return dataloader
        
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
