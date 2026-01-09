#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:01:11 2023

@author: ckervazo
"""
import torch
import torch.nn as nn
from torch.nn.functional import normalize
from code_christophe.munkres import Munkres
import numpy as np
#%%
def correctPerm_torch(W0_tc,W_tc):
    # [WPerm,Jperm,err] = correctPerm(W0,W)
    # Correct the permutations between the tensor W0_tc, W_tc
    # Usage : W_tc_perm_2 = torch.zeros_like(W_tc)
    #         for ii in range(W_tc.size()[0]):
    #               W_tc_perm_2[ii,:,:] = W_tc[ii,:,indTab[ii,:]]

    W_tc_corr_norm = torch.zeros_like(W_tc)
    W_tc_corr = torch.zeros_like(W_tc)
    indTab = torch.zeros((W_tc.size()[0],W_tc.size()[2])) # Premier indice : mini-batch, deuxieme : nombre de sources
    
    for ii in range(W0_tc.size()[0]):
        W0 = normalize(W0_tc[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        W = normalize(W_tc[ii,:,:],p=2.0,dim=0) # Une seule matrice (plus de mini-batch)
        
        W0 = W0.detach().cpu().numpy()
        W = W.detach().cpu().numpy()
        
        costmat = -W0.T@W; # Avec Munkres, il faut bien un -
    
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        WPerm = np.zeros(np.shape(W0))
        WPerm_norm = np.zeros(np.shape(W0))
        indPerm = np.zeros(np.shape(W0)[1])
        
        for jj in range(W0.shape[1]):
            WPerm[:,jj] = W_tc[ii,:,Jperm[jj][1]].detach().cpu().numpy()
            WPerm_norm[:,jj] = W[:,Jperm[jj][1]]
            indPerm[jj] = Jperm[jj][1]
        
        W_tc_corr_norm[ii,:,:] = torch.from_numpy(WPerm_norm)
        W_tc_corr[ii,:,:] = torch.from_numpy(WPerm)
        indTab[ii,:] = torch.from_numpy(indPerm)
    indTab = indTab.type(torch.int64)
    return W_tc_corr,W_tc_corr_norm,indTab

#%%
class SADLoss(nn.Module):
    # Ecrit pour les matrices de melanges. Transposer les deux entrees pour utiliser pour S. Ne corrige pas les permutations !
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self,targets,predictions):
        targets_norm = normalize(targets,p=2.0,dim=1)
        predictions_norm = normalize(predictions,p=2.0,dim=1)
        matConfusion = torch.bmm(torch.transpose(targets_norm, 1, 2),predictions_norm)
        
        diagBatch = torch.diagonal(matConfusion,dim1=1,dim2=2)# Prend la diagonale pour chaque mini-batch
        
        return -torch.sum(diagBatch)/(targets.size()[0]*targets.size()[2]) # Independant de la taille des mini-batchs et du nombre de sources
    
#%%
class toutesLoss(nn.Module):
    # Ecrit pour les matrices de melanges. Transposer les deux entrees pour utiliser pour S. Ne corrige pas les permutations !
    # 4 a utiliser : SAD sur A et NMSE sur S, 6 : seulement NMSE sur S
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
        
    def forward(self,A,A_pred,S,S_pred):
        if self.optLoss==1:
            train_A = self.criterion(A,A_pred)
            train_S = self.criterion(torch.transpose(S, 1, 2),torch.transpose(S_pred, 1, 2))
            
            train_loss = train_A + train_S
            
        elif self.optLoss==0:
            train_A = self.criterion(A,A_pred)
            train_S = self.criterion(S,S_pred)
            
            train_loss = train_A + train_S
            
        elif self.optLoss==2:
            A_norm = normalize(A,p=2.0,dim=1)
            A_pred_norm = normalize(A_pred,p=2.0,dim=1)
            S_norm = normalize(S,p=2.0,dim=2)
            S_pred_norm = normalize(S_pred,p=2.0,dim=2)

            train_A = self.criterion(A_norm,A_pred_norm)
            train_S = self.criterion(S_norm,S_pred_norm)
            
            train_loss = train_A + train_S

        elif self.optLoss==3:
            train_A = self.critSAD(A,A_pred)
            train_S = 1000*self.critMSE(S,S_pred)
            
            train_loss = train_A + train_S
            
        elif self.optLoss==4:
            train_A = self.critSAD(A,A_pred)
            train_S = self.critMSE(S,S_pred)/(torch.norm(S)**2)
            
            train_loss = train_A + train_S
            
        elif self.optLoss==5:
            train_A = self.critSAD(A,A_pred)
            train_S = 0
            
            for ii in range(S.size()[1]):
                train_S += self.critMSE(S[:,ii,:],S_pred[:,ii,:])/(torch.norm(S[:,ii,:])**2)
            
            train_loss = train_A + train_S
            
        elif self.optLoss==6:# Pas de loss sur 1, seulement sur S
            train_S = self.critMSE(S,S_pred)/(torch.norm(S)**2)
            train_A = torch.zeros(1)
            
            train_loss = train_S
            
        return train_loss,train_A,train_S
        
        
