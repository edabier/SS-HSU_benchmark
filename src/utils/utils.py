import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data as Data
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
#         targets_norm = F.normalize(targets,p=2.0,dim=1)
#         predictions_norm = F.normalize(predictions,p=2.0,dim=1)
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

def order_endmembers(E_gt, E_hat, A_hat=None):
    if E_hat.dim() == 2:
        E_hat = E_hat.unsqueeze(0)
    if E_gt.dim() == 2:
        E_gt = E_gt.unsqueeze(0)
        
    if A_hat is not None:
        if A_hat.dim() < 3:
            A_hat = A_hat.unsqueeze(0)
        A_hat_corr = torch.zeros_like(A_hat)
    
    E_hat_corr_norm = torch.zeros_like(E_hat)
    E_hat_corr = torch.zeros_like(E_hat)
    indices = torch.zeros((E_hat.size()[0],E_hat.size()[2])) # Premier indice : mini-batch, deuxieme : nombre de sources
    
    for ii in range(E_gt.size()[0]):
        E0 = F.normalize(E_gt[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        E = F.normalize(E_hat[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        E0 = E0.to(torch.float64)   
        E = E.to(torch.float64)
        
        costmat = -E0.T@E; # Avec Munkres, il faut bien un -
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        EPerm = torch.zeros(E0.shape)
        EPerm_norm = torch.zeros(E0.shape)
        perm_indices = torch.zeros(E0.shape[1])
    
        if A_hat is not None:
            APerm = torch.zeros(A_hat[ii].shape)
        
        for jj in range(E0.shape[1]):
            EPerm[:,jj] = E_hat[ii,:,Jperm[jj][1]]
            EPerm_norm[:,jj] = E[:,Jperm[jj][1]]
            perm_indices[jj] = Jperm[jj][1]
            if A_hat is not None:
                APerm[jj] = A_hat[ii, Jperm[jj][1]]
        
        E_hat_corr_norm[ii,:,:] = EPerm_norm
        E_hat_corr[ii,:,:] = EPerm
        indices[ii,:] =perm_indices
        if A_hat is not None:
            A_hat_corr[ii] = APerm
    indices = indices.type(torch.int64)

    if A_hat is not None:
        return E_hat_corr, E_hat_corr_norm, A_hat_corr, indices
    else:
        return E_hat_corr,E_hat_corr_norm,indices

def compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, name=None, use_wandb=False):
    """
    Computes the SAD of predicted E and MSE of predicted A
    """
    sad = SADLoss()
    mse = nn.MSELoss(reduction = "sum")

    num_E = e_hat.shape[-1]
    n = num_E // 2
    if num_E % 2 != 0: n = n + 1
    
    e_hat = e_hat/ e_hat.max(dim=0, keepdim=True).values
    e_gt = e_gt/ e_gt.max(dim=0, keepdim=True).values
    
    E_ordered, E_ordered_norm, a_hat, indices = order_endmembers(e_gt, e_hat, a_hat)
    E_ordered = E_ordered[0]
    a_hat = a_hat[0]

    if e_gt.dim() == 3:
        e_gt = e_gt[0]
    
    sads = []
    mses = []

    for i in range(num_E):
        sad_ = sad(e_gt[:, i], E_ordered[:, i])
        mse_ = mse(a_gt, a_hat)/(torch.norm(a_gt)**2)
        sads.append(sad_.item())
        mses.append(mse_.item())
    sads = torch.tensor(sads)
    mses = torch.tensor(mses)
    Average_SAD = torch.mean(sads)
    Average_MSE = torch.mean(mses)

    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
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
        plt.savefig(f"/home/ids/edabier/HSU/SS-HSU_benchmark/{name}_Es.png")

    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        # print("Plotting E image on wandb")
        wandb.log({"Endmember extraction": wandb.Image(img_array)})

    fig, axes = plt.subplots(a_hat.shape[0], 2, figsize=(5, 10))
    title = f"{name} aMSE score for all A: " + format(Average_MSE, '.5f')
    plt.suptitle(title)
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
        plt.savefig(f"/home/ids/edabier/HSU/SS-HSU_benchmark/{name}_As.png")

    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        # print("Plotting A image on wandb")
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

def normalize(Y, dim=0):
    """
    Normalizes the input tensor along the given dimension
    """
    shape = Y.shape
    max_values = torch.amax(torch.abs(Y), dim=[d for d in range(Y.dim()) if d != dim], keepdim=True)
    max_values[max_values == 0] = 1
    Y_normalized = Y / max_values

    return Y_normalized

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

        self.Y_2d, _ = oneD_to_2d(self.Y) # (B, H, W)
        self.A_2d, _ = oneD_to_2d(self.A) # (c, H, W)
        
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
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, 
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
