#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:12:07 2023

@author: ckervazo
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from train_LMU_checkpoint_CK import train
import torch
from munkres import Munkres
import scipy.io as sio
from small_networks import convolutionalNN_2D # A voir si vraiment utile
import os


#%%
folderInitSave = 'Results/test_DL_trainSum1_optSAD6_Apex_RALMU'
datasetNumber = 9 # 0 : natural images for chandra training, 1: chandra, 2: Samson avec Dirichlets, 3: Samson avec NNLS, 4: Jasper NNLS, 5 : Apex, 6 : donnees labos avec entrainement auto-supervise, 7 : donnees labos avec entrainement semi-supervise, 8 : spectro gamma jonathan, 9 : Apex dead leaves

if datasetNumber == 1:
    sizeHimage = [4,346,346]
elif datasetNumber == 4:
    sizeHimage = [4,100,100]
elif datasetNumber == 5 or datasetNumber == 9:
    sizeHimage = [4,110,110]
elif datasetNumber == 6 or datasetNumber == 7:
    sizeHimage = [3,60,60]
elif datasetNumber == 8:
    sizeHimage = [9]
else:
    sizeHimage = [3,95,95]
    
    
    
#%%
if torch.cuda.is_available():
    dev = "cuda:0"
    torch.set_default_device(dev)
else:
    dev = "cpu"


print(dev)


#%%
class My_dataset(Dataset):
    def __init__(self, V, W, H):
        self.V = V.copy()
        self.W = W.copy()
        self.H = H.copy()
        
    def __len__(self):
        return np.shape(self.V)[0]

    def __getitem__(self, idx):
        
        return self.V[idx,:,:], self.W[idx,:,:], self.H[idx,:,:]

if datasetNumber == 8:
    data = np.load('Donnees_spectro_Jonathan/data_christophe_withGTtest.npz')
    dataDict= dict(zip(data.files, (data[f] for f in data.files)))
    
#%% Pour la lecture du jeu de donnees d'entrainement
class My_large_training_dataset(Dataset):
    def __init__(self):
        print('Initialisation')
        
    def __len__(self):
        if datasetNumber < 8:
            return 750
        elif datasetNumber == 8:
            return 15000
        elif datasetNumber == 9:
            return 750

    def __getitem__(self, idx):
        if datasetNumber == 0:
            V = torch.tensor(sio.loadmat('training_set_imNaturelles_small_NN_scale/V_%s'%idx)['donnees'])
            W = torch.tensor(sio.loadmat('training_set_imNaturelles_small_NN_scale/W_%s'%idx)['donnees'])
            H = torch.tensor(sio.loadmat('training_set_imNaturelles_small_NN_scale/H_%s'%idx)['donnees'])
        elif datasetNumber == 1:
            V = torch.tensor(sio.loadmat('training_set_chandra/V_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('training_set_chandra/W_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('training_set_chandra/H_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 2:
            V = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Train_set/X_dirichlet_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Train_set/S_dirichlet_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 3:
            V = torch.tensor(sio.loadmat('Donnees_Samson_CK/Train_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_CK/Train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_CK/Train_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        
        elif datasetNumber == 4:
            V = torch.tensor(sio.loadmat('Donnees_Jasper_CK/train_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Jasper_CK/train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Jasper_CK/train_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        
        elif datasetNumber == 5:
            V = torch.tensor(sio.loadmat('Donnees_Apex_CK/train_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Apex_CK/train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Apex_CK/train_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 6:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 7:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set_semi_supervised/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set_semi_supervised/A_GT_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/train_set_semi_supervised/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        elif datasetNumber == 8:
            V = torch.tensor(dataDict['trainX'][idx,:], dtype=torch.float32).unsqueeze(1)
            W = torch.tensor(dataDict['trainS'][idx,:,:], dtype=torch.float32)
            H = torch.tensor(dataDict['trainA'][idx,:], dtype=torch.float32).unsqueeze(1)
            
        elif datasetNumber == 9:
            V = torch.tensor(sio.loadmat('Donnees_Apex_DL_sum1/train_set/X_DL_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Apex_CK/train_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32) # Meme que dataset 5
            H = torch.tensor(sio.loadmat('Donnees_Apex_DL_sum1/train_set/DL_4_%s.mat'%idx)['donnees']).to(torch.float32)
        return V,W,H
    
#%%
train_set = My_large_training_dataset()

#%% Pour la lecture des donnees
class My_large_validation_dataset(Dataset):
    def __init__(self):
        print('Initialisation')
        
    def __len__(self):
        if datasetNumber < 8:
            return 150
        elif datasetNumber == 8:
            return 1500
        elif datasetNumber == 9:
            return 150

    def __getitem__(self, idx):
        if datasetNumber == 0:
            V = torch.tensor(sio.loadmat('validation_set_imNaturelles_small_NN_scale/V_%s'%idx)['donnees']).to(dev)
            W = torch.tensor(sio.loadmat('validation_set_imNaturelles_small_NN_scale/W_%s'%idx)['donnees']).to(dev)
            H = torch.tensor(sio.loadmat('validation_set_imNaturelles_small_NN_scale/H_%s'%idx)['donnees']).to(dev)
        elif datasetNumber == 1:
            V = torch.tensor(sio.loadmat('validation_set_chandra/V_%s'%idx)['donnees']).to(dev).to(torch.float32)
            W = torch.tensor(sio.loadmat('validation_set_chandra/W_%s'%idx)['donnees']).to(dev).to(torch.float32)
            H = torch.tensor(sio.loadmat('validation_set_chandra/H_%s'%idx)['donnees']).to(dev).to(torch.float32)
        elif datasetNumber == 2:
            V = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Validation_set/X_dirichlet_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Validation_set/S_dirichlet_%s'%idx)['donnees']).to(torch.float32)
        elif datasetNumber == 3:
            V = torch.tensor(sio.loadmat('Donnees_Samson_CK/Validation_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_CK/Validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_CK/Validation_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        
        elif datasetNumber == 4:
            V = torch.tensor(sio.loadmat('Donnees_Jasper_CK/validation_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Jasper_CK/validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Jasper_CK/validation_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 5:
            V = torch.tensor(sio.loadmat('Donnees_Apex_CK/validation_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Apex_CK/validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Apex_CK/validation_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        
        elif datasetNumber == 6:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
            
        elif datasetNumber == 7:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set_semi_supervised/X_NNLS_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set_semi_supervised/A_GT_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/validation_set_semi_supervised/S_NNLS_%s'%idx)['donnees']).to(torch.float32)
        elif datasetNumber == 8:
            V = torch.tensor(dataDict['valX'][idx,:], dtype=torch.float32).unsqueeze(1) + 1e-4
            W = torch.tensor(dataDict['valS'][idx,:,:], dtype=torch.float32) + 1e-4
            H = torch.tensor(dataDict['valA'][idx,:], dtype=torch.float32).unsqueeze(1) + 1e-4
        
        elif datasetNumber == 9:
            V = torch.tensor(sio.loadmat('Donnees_Apex_DL_sum1/validation_set/X_DL_%s'%idx)['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Apex_CK/validation_set/A_VCA_L1_%s'%idx)['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Apex_DL_sum1/validation_set/DL_4_%s.mat'%idx)['donnees']).to(torch.float32)
        
        return V,W,H   
#%%
val_set = My_large_validation_dataset()

#%% Pour la lecture des donnees
class My_testing_dataset(Dataset):
    def __init__(self):
        print('Initialisation')
        
    def __len__(self):
        if datasetNumber == 0 or datasetNumber == 1:
            return 150
        elif datasetNumber >= 2 and datasetNumber < 8:
            return 20
        elif datasetNumber == 8:
            return 40000
        elif datasetNumber == 9:
            return 40
        
    def __getitem__(self, idx):
        if datasetNumber == 0 or datasetNumber == 1:
            V = torch.tensor(sio.loadmat('testing_set_NN_scale/V_%s'%idx)['donnees']).to(dev).to(torch.float32)
            W = torch.tensor(sio.loadmat('testing_set_NN_scale/W_%s'%idx)['donnees']).to(dev).to(torch.float32)
            H = torch.tensor(sio.loadmat('testing_set_NN_scale/H_%s'%idx)['donnees']).to(dev).to(torch.float32)
        if datasetNumber == 2:
            V = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Test_set/X_normalized_l1norm.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Test_set/A_samson_normalized_l1norm.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_Rassim_l1_norm/Test_set/S_samson.mat')['S_samson']).to(torch.float32)
        if datasetNumber == 3:
            V = torch.tensor(sio.loadmat('Donnees_Samson_CK/Test_set/X_normalized_l1norm.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Samson_CK/Test_set/A_samson_normalized_l1norm.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Samson_CK/Test_set/S_samson.mat')['S_samson']).to(torch.float32)
                
        if datasetNumber == 4:
            V = torch.tensor(sio.loadmat('Donnees_Jasper_CK/test_set/X_normalized_l1norm.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Jasper_CK/test_set/A_jasper_normalized_l1norm.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Jasper_CK/test_set/S_jasper.mat')['donnees']).to(torch.float32)
            
        if datasetNumber == 5 or datasetNumber == 9:
            V = torch.tensor(sio.loadmat('Donnees_Apex_CK/test_set/X_Apex_normalized_l1.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('Donnees_Apex_CK/test_set/A_Apex_normalized_l1.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('Donnees_Apex_CK/test_set/S_Apex.mat')['donnees']).to(torch.float32)
            
        if datasetNumber == 6:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/X_Sc2Mix4_normalized_l1norm.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/A_Scene2Mix4_normalized_l1norm.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/S_Sc2Mix4.mat')['donnees']).to(torch.float32)
            
        if datasetNumber == 7:
            V = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/X_Sc2Mix4_normalized_l1norm.mat')['donnees']).to(torch.float32)
            W = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/A_Scene2Mix4_normalized_l1norm.mat')['donnees']).to(torch.float32)
            H = torch.tensor(sio.loadmat('DonneesLaboScene2Mix4/test_set/S_Sc2Mix4.mat')['donnees']).to(torch.float32)
        
        if datasetNumber == 8:
            V = torch.tensor(dataDict['testX'][idx,:], dtype=torch.float32).unsqueeze(1) + 1e-4
            W = torch.tensor(dataDict['testS'][idx,:,:], dtype=torch.float32) + 1e-4
            H = torch.tensor(dataDict['testA'][idx,:], dtype=torch.float32).unsqueeze(1) + 1e-4

        return V,W,H

#%%
test_set = My_testing_dataset()

#%% Parameters to be set
batch_size = 10 # 1499 pour spectro
num_epochs = 3000
T = 25
params_shared = False # For tied weights
lr = 0.00001 #0.0001 pour spectro
optSAD = 4
optNN = 1 # 1 RALMU, 0 NALMU
sizeConv = 5 # 5, -1 si pas conv. Inactif si pas NN ou si UNet

#%%
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last = True, generator=torch.Generator(device=dev)) # train = images naturelles pour abondances
val_loader = DataLoader(val_set,batch_size=batch_size , shuffle=True, drop_last = True, generator=torch.Generator(device=dev)) # Validation = images naturelles pour abondances
test_loader = DataLoader(test_set,batch_size=batch_size , shuffle=True, drop_last = True, generator=torch.Generator(device=dev)) # Validation = images naturelles pour abondances

#%%
train_total_loss, val_total_loss,model,test_total_loss,train_total_A_loss_disp,train_total_S_loss_disp,val_total_A_loss_disp,val_total_S_loss_disp,test_total_A_loss_disp,test_total_S_loss_disp= train(train_loader, val_loader, test_loader, num_epochs=num_epochs, T=T,params_shared=params_shared,lr = lr,folderInitSave = folderInitSave, optSAD=optSAD,optNN=optNN,sizeConv=sizeConv,sizeHimage=sizeHimage)
#%%
torch.save(model,folderInitSave + '/LMU_model.pth')
torch.save(val_total_loss,folderInitSave + '/LMU_val_total_loss.pth')
torch.save(train_total_loss,folderInitSave + '/LMU_train_total_loss.pth')
torch.save(test_total_loss,folderInitSave + '/LMU_test_total_loss.pth')
torch.save(train_total_A_loss_disp,folderInitSave + '/LMU_train_total_A_loss_disp.pth')
torch.save(train_total_S_loss_disp,folderInitSave + '/LMU_train_total_S_loss_disp.pth')
torch.save(val_total_A_loss_disp,folderInitSave + '/LMU_val_total_A_loss_disp.pth')
torch.save(val_total_S_loss_disp,folderInitSave + '/LMU_val_total_S_loss_disp.pth')
torch.save(test_total_A_loss_disp,folderInitSave + '/LMU_test_total_A_loss_disp.pth')
torch.save(test_total_S_loss_disp,folderInitSave + '/LMU_test_total_S_loss_disp.pth')
