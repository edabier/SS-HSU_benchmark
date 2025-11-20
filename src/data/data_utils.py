import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as io

if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    dev = "cpu"
    
class HSI_dataset(Dataset):
    """
    Defines a HSI dataset object
    
    Args:
        dataset (str): the name of the dataset to load (must be: `samson`, `jasper`, `urban`, `apex`, or `simulee_1,2,3`)
    """
    def __init__(self, dataset):

        data_path = "datasets/" + dataset + ".mat"
        data = io.loadmat(data_path)
        
        if dataset == 'samson':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 95
        elif dataset == 'jasper':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 100
        elif dataset == 'urban':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 307
        elif dataset == 'apex':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 110
        elif dataset == 'simulee_1':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 75
        elif dataset == 'simulee_2':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 100
        elif dataset == 'simulee_3':
            self.c, self.B, self.col = data['E'].shape[0], data['E'].shape[1], 105
        
        self.Y = torch.from_numpy(data['Y'])
        self.A = torch.from_numpy(data['A'])
        self.E = torch.from_numpy(data['E'])
        
    def __len__(self):
        return np.shape(self.Y)[0]

    def __getitem__(self, idx):
        return self.Y[idx,:,:], self.E[idx,:,:], self.A[idx,:,:]