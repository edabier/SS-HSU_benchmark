import torch
import torch.nn as nn
from torch.nn.functional import normalize
from munkres import Munkres
import numpy as np

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
        
        W0 = W0.detach().numpy()
        W = W.detach().numpy()
        
        costmat = -W0.T@W; # Avec MunkreA, il faut bien un -
    
        
        m = Munkres()
        Jperm = m.compute(costmat.tolist())
        
        WPerm = np.zeros(np.shape(W0))
        WPerm_norm = np.zeros(np.shape(W0))
        indPerm = np.zeros(np.shape(W0)[1])
        
        for jj in range(W0.shape[1]):
            WPerm[:,jj] = W_tc[ii,:,Jperm[jj][1]].detach().numpy()
            WPerm_norm[:,jj] = W[:,Jperm[jj][1]]
            indPerm[jj] = Jperm[jj][1]
        
        W_tc_corr_norm[ii,:,:] = torch.from_numpy(WPerm_norm)
        W_tc_corr[ii,:,:] = torch.from_numpy(WPerm)
        indTab[ii,:] = torch.from_numpy(indPerm)
    indTab = indTab.type(torch.int64)
    return W_tc_corr,W_tc_corr_norm,indTab


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