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

def CNNAEU_loss(target, pred):
    assert target.shape == pred.shape

    dot_product = (target * pred).sum(dim=1)
    target_norm = target.norm(dim=1)
    pred_norm = pred.norm(dim=1)
    sad_score = torch.clamp(dot_product / (target_norm * pred_norm), -1, 1).acos()
    return sad_score.mean()

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
 
def numpy_SAD(y_true, y_pred):
    return np.arccos(np.dot(y_pred, y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred)))

def alter_MSE(y_true, y_pred):
    num_em = y_true.shape[0]
    y_true = np.reshape(y_true , [num_em, -1])
    y_pred = np.reshape(y_pred , [num_em, -1])

    R = y_pred - y_true
    r = R*R
    mse = np.mean(r, axis=1)
    Average_mse = np.sum(mse) / len(mse)
    mse = np.insert(mse, num_em, Average_mse, axis=0)
    return mse

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
    
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.99, last_epoch=-1):
        
        self.max_iter = max_iter
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * (1 - self.last_epoch / self.max_iter) ** self.power
            for base_lr in self.base_lrs
        ]
    
class ReduceLREveryNEpochs(_LRScheduler):
    def __init__(self, optimizer, reduce_every=15, reduce_by=0.2, last_epoch=-1):
        self.reduce_every = reduce_every
        self.reduce_by = reduce_by
        super(ReduceLREveryNEpochs, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > 0 and self.last_epoch % self.reduce_every == 0:
            return [lr * (1 - self.reduce_by) for lr in self.base_lrs]
        else:
            return self.base_lrs

class HSI_dataset(Dataset):    
    def __init__(self, dataset, patch_size=None, dtype=None):
        
        self.dataset_name = dataset
        
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
   
def create_dataloader(dataset, dev, train_split=None, patch_size=None, batch_size=1, dtype=torch.float32):
    """
    Creates dataloader(s) for a given dataset
    
    Args:
        dataset (str): the name of the dataset to use
        dev: the device on which to pass the loaders
        train_split (float, optional): how much of the dataset to use for the training and testing sets
        patch_size (int): whether or not to patch the input HSI
    """
    if patch_size is None:
        dataset = HSI_dataset(dataset, dtype=dtype)
    else:
        dataset = HSI_dataset(dataset, patch_size, dtype=dtype)
        
    if train_split is not None:
        generator = torch.Generator(dev)
        train_set, test_set = random_split(dataset, lengths=[train_split, 1-train_split], generator=generator)

        train_loader = DataLoader(train_set, batch_size)
        test_loader = DataLoader(test_set, batch_size)
        return train_loader, test_loader, dataset.B, dataset.col
    else:
        train_loader = DataLoader(dataset, batch_size)
        return train_loader, dataset.B, dataset.col
            
def save_model(model, optimizer, directory, name, epoch, is_permanent=False):
    """
    Overwrite the previous checkpoint save if not is_permanent, otherwise, saves a new version of the model
    """
    if is_permanent:
        # Save a permanent copy of the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(directory, f'{model.__class__.__name__}_{name}_lr_{optimizer.param_groups[-1]["lr"]}_epoch_{epoch}.pt'))
        print(f'Saved permanent model {model.__class__.__name__}_{name}_lr_{optimizer.param_groups[-1]["lr"]}_epoch_{epoch}.pt')
    else:
        # Overwrite the temporary model save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(directory, f'{model.__class__.__name__}_{name}_lr_{optimizer.param_groups[-1]["lr"]}.pt'))
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

def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    dict = {}
    SAD_ = []
    sad_mat = np.ones((num_endmembers, num_endmembers))
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
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

def compute_metrics_and_plot(e_hat, e_gt, a_hat, a_gt, dataset_name=None):
    """
    Computes the SAD of predicted E and MSE of predicted A
    """
    
    e_hat = e_hat.T
    e_gt = e_gt.T

    num_E = e_hat.shape[0]
    n = num_E // 2
    if num_E % 2 != 0: n = n + 1

    dict, _, Average_SAM = order_endmembers(e_hat.cpu().numpy(), e_gt.cpu().numpy())
    sad_ordered = []
    endmember_ordered = []
    abundance_ordered = []

    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = f"{dataset_name} aSAM score for all E: " + format(Average_SAM, '.3f') + " radians"
    st = plt.suptitle(title)
    for i in range(num_E):
        e_hat[i, :] = e_hat[i, :] / e_hat[i, :].max()
        e_gt[i, :] = e_gt[i, :] / e_gt[i, :].max()

    for i in range(num_E):
        endmember_ordered.append(e_hat[dict[i]].cpu())
        abundance_ordered.append(a_hat[dict[i], :, :].detach().cpu())
    endmember_ordered = np.array(endmember_ordered)
    for i in range(num_E):
        z = numpy_SAD(endmember_ordered[i], e_gt[i, :].cpu())
        sad_ordered.append(z)

    for i in range(num_E):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(e_gt[i, :].cpu(), 'r', linewidth=1.0, label='GT')
        plt.plot(endmember_ordered[i, :], 'k-', linewidth=1.0, label='predict')
        plt.legend()
        ax.set_title("SAD: " + format(sad_ordered[i], '.4f'))
        ax.get_xaxis().set_visible(False)
        
    sad_ordered.append(Average_SAM)
    sad_ordered = np.array(sad_ordered)

    abundance_ordered = np.array(abundance_ordered)

    mse = alter_MSE(a_gt.cpu().numpy(), abundance_ordered)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)

    fig, axes = plt.subplots(a_hat.shape[0], 2, figsize=(5, 10))
    axes[0, 0].set_title("Abundance pred", fontsize=12)
    axes[0, 1].set_title("Abundance GT", fontsize=12)

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

    return mse, sad_ordered

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