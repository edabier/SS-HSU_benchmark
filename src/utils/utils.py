import torch
import torch.nn as nn
from torch.nn.functional import normalize
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
from scipy.optimize import linear_sum_assignment
from code_christophe.munkres import Munkres

# class SADLoss(nn.Module):
#     """
#     SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
#     """
#     def __init__(self):
#         super(SADLoss, self).__init__()

#     def forward(self, targets, predictions):
#         targets_norm = normalize(targets,p=2.0,dim=1)
#         predictions_norm = normalize(predictions,p=2.0,dim=1)
#         matConfusion = torch.bmm(torch.transpose(targets_norm, 1, 2),predictions_norm)
        
#         diagBatch = torch.diagonal(matConfusion,dim1=1,dim2=2)
        
#         return -torch.sum(diagBatch)/(targets.size()[0]*targets.size()[2])

# class SADTrans(nn.Module):
#     def __init__(self, num_bands):
#         super(SAD, self).__init__()
#         self.num_bands = num_bands

#     def forward(self, inp, target):
#         try:
#             input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
#                                               inp.view(-1, self.num_bands, 1)))
#             target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
#                                                target.view(-1, self.num_bands, 1)))

#             summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
#             angle = torch.acos(summation / (input_norm * target_norm))

#         except ValueError:
#             return 0.0

#         return angle

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = torch.nn.functional.normalize(y_true, dim=0, p=2)
            y_pred = torch.nn.functional.normalize(y_pred, dim=0, p=2)
        else:
            y_true = torch.nn.functional.normalize(y_true, dim=1, p=2)
            y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss

# class SADLoss(nn.Module):
#     """
#     SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
#     """
#     def __init__(self):
#         super(SADLoss, self).__init__()

#     def forward(self, y_true, y_pred):

#         assert y_true.shape == y_pred.shape

#         dot_product = (y_true * y_pred).sum(dim=1)
#         target_norm = y_true.norm(dim=1)
#         output_norm = y_pred.norm(dim=1)
#         sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
#         return sad_score.mean()

def numpy_SAD(y_true, y_pred):
    return np.cos(np.arccos(np.dot(y_pred, y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))))

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
   
def numpy_order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    SAD_ = []
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[:,i] = endmembers[:,i] / endmembers[:,i].max()
        endmembersGT[:,i] = endmembersGT[:,i] / endmembersGT[:,i].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i,:], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if minimum == 100:
            break
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        dict[index[1]] = index[0]  # keep Gt at first,
        SAD_.append(minimum)
        # dict[index[0]] = index[1]
        sad_mat[index[0], index[1]] = 100
        rows += 1
        sad_mat[index[0], :] = 100
        sad_mat[:, index[1]] = 100
    SAD_ = np.array(SAD_)
    Average_SAM = np.sum(SAD_)/ len(SAD_)
    return dict, SAD_, Average_SAM

def order_endmembers(E_hat, E_gt):

    E_hat_corr_norm = torch.zeros_like(E_hat)
    E_hat_corr = torch.zeros_like(E_hat)
    indTab = torch.zeros((E_hat.size()[0],E_hat.size()[2])) # Premier indice : mini-batch, deuxieme : nombre de sources
    
    for ii in range(E_gt.size()[0]):
        W0 = normalize(E_gt[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        W = normalize(E_hat[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        
        W0 = W0.detach().cpu().numpy()
        W = W.detach().cpu().numpy()
        
        costmat = -W0.T@W; # Avec Munkres, il faut bien un -
    
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        WPerm = np.zeros(np.shape(W0))
        WPerm_norm = np.zeros(np.shape(W0))
        indPerm = np.zeros(np.shape(W0)[1])
        
        for jj in range(W0.shape[1]):
            WPerm[:,jj] = E_hat[ii,:,Jperm[jj][1]].detach().cpu().numpy()
            WPerm_norm[:,jj] = W[:,Jperm[jj][1]]
            indPerm[jj] = Jperm[jj][1]
        
        E_hat_corr_norm[ii,:,:] = torch.from_numpy(WPerm_norm)
        E_hat_corr[ii,:,:] = torch.from_numpy(WPerm)
        indTab[ii,:] = torch.from_numpy(indPerm)
    indTab = indTab.type(torch.int64)

    return E_hat_corr,E_hat_corr_norm,indTab

def compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=None, use_wandb=False):
    """
    Computes the SAD of predicted E and MSE of predicted A
    """
    sad = SADLoss()
    mse = nn.MSELoss(reduction = "sum")

    num_E = e_hat.shape[-1]
    n = num_E // 2
    if num_E % 2 != 0: n = n + 1

    dict, _, _ = order_endmembers(e_hat, e_gt)

    E_ordered = []
    A_ordered = []
    
    for i in range(num_E):
        E_ordered.append(e_hat[:, dict[i]])
        A_ordered.append(a_hat[dict[i], :])
    
    E_ordered = torch.stack(E_ordered, dim=1)
    A_ordered = torch.stack(A_ordered, dim=0)
    # Average_MSE = mse(a_gt,A_ordered)/(torch.norm(a_gt)**2)

    dict, _, Average_MSE = order_abundances(a_hat, a_gt)

    E_ordered = []
    A_ordered = []
    for i in range(num_E):
        E_ordered.append(e_hat[:, dict[i]])
        A_ordered.append(a_hat[dict[i],:, :])
    E_ordered = torch.stack(E_ordered, dim=1)
    A_ordered = torch.stack(A_ordered, dim=0)

    sad_ordered = []

    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()

    for i in range(num_E):
        z = sad(e_gt[:, i], E_ordered[:, i])
        sad_ordered.append(z.item())
    sad_ordered = torch.tensor(sad_ordered)

    Average_SAD = torch.mean(sad_ordered)
    title = f"{name} aSAD score for all E: " + format(Average_SAD, '.5f')
    st = plt.suptitle(title)

    E_ordered = E_ordered/ E_ordered.max(dim=0, keepdim=True).values
    e_gt = e_gt/ e_gt.max(dim=0, keepdim=True).values

    for i in range(num_E):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(e_gt[:, i].detach().cpu(), 'r', linewidth=1.0, label='GT')
        plt.plot(E_ordered[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
        plt.legend()
        ax.set_title("SAD: " + format(sad_ordered[i], '.5f'))
        ax.get_xaxis().set_visible(False)
        
    # sad_ordered.append(Average_SAM)
    # sad_ordered = torch.stack(sad_ordered)

    # mse = alter_MSE(a_gt.detach().cpu().numpy(), A_ordered.cpu().numpy())
    # mse_scores = []
    # for i in range(num_E):
    #     z = mse(a_gt[:, i], A_ordered[:, i])
    #     mse_scores.append(z.item())
    # mse_scores = torch.tensor(mse_scores)
    # Average_MSE = torch.mean(mse_scores)

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

def compute_metrics(E, A, E_hat, A_hat, rmse=False):
    
    if rmse:
        re = torch.mean(torch.sqrt(torch.mean((A - A_hat) ** 2, dim=2)))
    else:
        re = torch.mean(torch.sum((A - A_hat) ** 2, dim=2))
        
    E_norm = E / torch.norm(E, dim=0, keepdim=True)
    E_hat_norm = E_hat / torch.norm(E_hat, dim=0, keepdim=True)
    sad = torch.mean(torch.acos(torch.clamp(torch.sum(E_norm * E_hat_norm, dim=0), -1.0, 1.0)))
    
    return re, sad

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
        Y, _ = oneD_to_2d(Y)
        s = k*patch_size
        Y = Y[:, :, :s, :s]
        Y = Y.reshape(batch, B, s**2)
        
        if A is not None:
            A, _ = oneD_to_2d(A)
            A = A[:, :, :s, :s]
            A = A.reshape(batch, A.shape[1], s**2)
            return Y, A
        else:
            return Y
    else:
        Y, _ = oneD_to_2d(Y)
        s = k*patch_size
        Y = Y[:, :s, :s]
        Y = Y.reshape(B, s**2)
        
        if A is not None:
            A, _ = oneD_to_2d(A)
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
        return Y.reshape(batch, B, H, H), H
    else:
        B, N = Y.shape
        H = int(N**0.5)
        return Y.reshape(B, H, H), H

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
