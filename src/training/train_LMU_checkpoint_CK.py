import torch
from NALMU import NALMU
from RALMU import LMU_2nd_reseau as RALMU
import numpy as np
import os
from utils import toutesLoss
from small_networks import convolutionalNN_2D

# To enable SLURM restart
def save_checkpoint(data, filename):
    # Sauvegarde des données d'état
    torch.save(data, filename)
    
def load_checkpoint(filename):
    # Chargement des données si elles existent
    if os.path.exists(filename):
        return torch.load(filename)
    return None
    
def train(train_loader, val_loader, test_loader, num_epochs=10, T=10,params_shared=False,lr = 0.0001,folderInitSave = '',optSAD=0,optNN = 0, sizeConv=3,sizeHimage=[4,346,346],tab_mlp_H=''):
    # optNN = 0 : matrice Aw fixe, optNN = 1 : une reestimation avec un NN (optNN = True dans version precedente), optNN = i : pour la ieme reestimation
    
    coeffsLossAllLayer = np.linspace(0.0, 1.0,num=T)
    print('Criterion %s'%optSAD)
    
    torch.autograd.set_detect_anomaly(True)
    
    criterion = toutesLoss(optLoss=optSAD)    
    
    A_init_disp = next(iter(train_loader))[1].to(torch.float32)
    S_init_disp = torch.ones(next(iter(train_loader))[2].size(),dtype=torch.float32)
    A_init = torch.ones_like(A_init_disp)
    S_init = torch.ones_like(S_init_disp)
    
    unTuple = next(iter(test_loader))
    A_init_test_disp = unTuple[1].to(torch.float32)
    S_init_test_disp = unTuple[2].to(torch.float32)
    A_init_test = torch.ones_like(A_init_test_disp)
    S_init_test = torch.ones_like(S_init_test_disp)    
    
    if optNN > 0:
        model = RALMU(T=T,m=A_init.size()[1],n=A_init.size()[2],params_shared = params_shared,sizeConv=sizeConv,sizeHimage=sizeHimage)
    else:
        model = NALMU(T=T,n=A_init.size()[2],m=A_init.size()[1],t=S_init.size()[2],params_shared = params_shared)
    
    total_params = [p.numel() for p in model.parameters()]
    print(total_params)
    
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=lr, betas=(0.9, 0.999))

    # To enable SLURM restart
    checkpoint = load_checkpoint(filename=folderInitSave + '/checkpoint.pth')
    if checkpoint:
        # Reprendre là où le script s'est arrêté
        nbEpochsPretrained = checkpoint['epoch']
        model_state = checkpoint['model_state']
        optimizer_state = checkpoint['optimizer_state']
        
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    else:
        # Initialiser depuis le début
        nbEpochsPretrained = 0
    # End slurm restart
        
    train_total_loss = []
    train_total_A_loss_disp = []
    train_total_S_loss_disp = []
    val_total_loss = []
    val_total_A_loss_disp = []
    val_total_S_loss_disp = []
    test_total_loss = []
    test_total_A_loss_disp = []
    test_total_S_loss_disp = []
    
    ################################ Epoch "-1" pour evaluer les metriques avant tout apprentissage 
    train_total = 0
    train_total_A_disp = 0
    train_total_S_disp = 0
    model.eval()
    with torch.no_grad():
        for i, (X,A,S) in enumerate(train_loader):
            if optNN == 1:
                S_pred_pretrained_init = S_init
                A_pred_pretrained_init = A_init
                S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)

            elif optNN == 0:
                S_pred_tab, A_pred_tab = model(X,H_init=S_init, W_init = A_init)
                
            
            train_loss,train_loss_A,train_loss_S = criterion(A,A_pred_tab[-1],S,S_pred_tab[-1])
            
            train_total += train_loss.item()
            train_total_A_disp += train_loss_A.item()
            train_total_S_disp += train_loss_S.item()
            
        train_total /= i+1 # On divise par le nombre de mini-batchs pour ne pas avoir de differences si il y a plus de donnees dans un jeu de donnes
        train_total_loss.append(train_total)# Average loss over the different mini-batches at a given iteration
        
        train_total_A_disp /= i+1
        train_total_A_loss_disp.append(train_total_A_disp)
        
        train_total_S_disp /= i+1
        train_total_S_loss_disp.append(train_total_S_disp)
    
        ##### Metriques validation
    
        val_total = 0
        val_total_A_disp = 0
        val_total_S_disp = 0
        for i, (X,A,S) in enumerate(val_loader):
            if optNN == 1:
                S_pred_pretrained_init = S_init
                A_pred_pretrained_init = A_init
                S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)
            elif optNN == 0:
                S_pred_tab, A_pred_tab = model(X,H_init=S_init,W_init=A_init)
                
               
            val_loss,val_loss_A,val_loss_S = criterion(A,A_pred_tab[-1],S,S_pred_tab[-1])
            
            val_total += val_loss.item()
            val_total_A_disp += val_loss_A.item()
            val_total_S_disp += val_loss_S.item()
        
        val_total /= i+1
        val_total_loss.append(val_total)
        
        val_total_A_disp /= i+1
        val_total_A_loss_disp.append(val_total_A_disp)
        
        val_total_S_disp /= i+1
        val_total_S_loss_disp.append(val_total_S_disp)
        
        ##### Metriques test
        test_total = 0
        test_total_A_disp = 0
        test_total_S_disp = 0
        for i, (X,A,S) in enumerate(test_loader):
            if optNN == 1:
                S_pred_pretrained_init = S_init_test
                A_pred_pretrained_init = A_init_test
                S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)
            elif optNN == 0:
                S_pred_tab, A_pred_tab = model(X,H_init=S_init_test,W_init=A_init_test)

                
            test_loss,test_loss_A,test_loss_S = criterion(A,A_pred_tab[-1],S,S_pred_tab[-1])
            
            test_total += test_loss.item()# Pas detach initialement
            test_total_A_disp += test_loss_A.item()
            test_total_S_disp += test_loss_S.item()
            
        test_total /= i+1
        test_total_loss.append(test_total)
        
        test_total_A_disp /= i+1
        test_total_A_loss_disp.append(test_total_A_disp)
        
        test_total_S_disp /= i+1
        test_total_S_loss_disp.append(test_total_S_disp)
        
    ############################### Fin epoch "-1", debut entrainement
    
    for epoch in range(nbEpochsPretrained,num_epochs):
        train_total = 0
        train_total_A_disp = 0
        train_total_S_disp = 0
        
        model.train()# Notamment, pour les batchnorms du UNet
        for i, (X,A,S) in enumerate(train_loader):
            print("Epoch %s, mini-batch %s"%(epoch,i))
            optimizer.zero_grad()
            
            
            if optNN == 1:
                S_pred_pretrained_init = S_init
                A_pred_pretrained_init = A_init
                S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)
                    # S_pred_tab, A_pred_tab = model(X,H_init=S_init,W_init=A_init,indMbDisp=i)
            
            elif optNN == 0:
                S_pred_tab, A_pred_tab = model(X,H_init=S_init,W_init=A_init)
            
            train_loss = 0
            for iT in range(T):
                temp_train_loss,train_loss_A,train_loss_S = criterion(A,A_pred_tab[iT],S,S_pred_tab[iT]) # Pour train_loss_A et train_loss_S, qui ne sont utilise que pour le display, on ne gardera que les valeurs pour le dernier layers (la metrique utile sera alors loss_A + loss_S, qui sera differente de la train loss qui sera une somme ponderee sur les layers)
                train_loss += coeffsLossAllLayer[iT]*temp_train_loss
            
            print('Epoch %s, minibatch %s : loss A %s, loss S %s'%(epoch,i,train_loss_A,train_loss_S))
            train_loss.backward()
            optimizer.step()
            
            train_total += train_loss.item()# Remarque : dans le cas ou on met des coeffs, on enregistre la somme des losses sur les layers ponderee par les coefficients dans coeffsLossAllLayer
            train_total_A_disp += train_loss_A.item()
            train_total_S_disp += train_loss_S.item()
                
            
        train_total /= i+1 # On divise par le nombre de mini-batchs pour ne pas avoir de differences si il y a plus de donnees dans un jeu de donnes
        train_total_loss.append(train_total)# Average loss over the different mini-batches at a given iteration
        
        train_total_A_disp /= i+1
        train_total_A_loss_disp.append(train_total_A_disp)
        
        train_total_S_disp /= i+1
        train_total_S_loss_disp.append(train_total_S_disp)
        
        torch.save(model,folderInitSave+'/model_enCours_mod%s.pth'%np.mod(epoch,3))
        torch.save(train_total_loss,folderInitSave+'/train_total_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
        torch.save(train_total_A_loss_disp,folderInitSave+'/train_total_A_loss_disp_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
        torch.save(train_total_S_loss_disp,folderInitSave+'/train_total_S_loss_disp_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
        
        if np.mod(epoch,50) == 0:
            torch.save(model,folderInitSave+'/model_enCours_mod%s.pth'%epoch)
        
        #torch.save(torch.zeros(2),folderInitSave+'/epoch%s'%epoch)
        
        model.eval()
        with torch.no_grad():
            val_total = 0
            val_total_A_disp = 0
            val_total_S_disp = 0
            for i, (X,A,S) in enumerate(val_loader):
                
                if optNN == 1:
                    S_pred_pretrained_init = S_init
                    A_pred_pretrained_init = A_init
                    S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)
                elif optNN == 0:
                    S_pred_tab, A_pred_tab = model(X,H_init=S_init,W_init=A_init)
                    
                val_loss,val_loss_A,val_loss_S = criterion(A,A_pred_tab[-1],S,S_pred_tab[-1])
                
                val_total += val_loss.item()
                val_total_A_disp += val_loss_A.item()
                val_total_S_disp += val_loss_S.item()
                
            val_total /= i+1
            val_total_loss.append(val_total)
            
            val_total_A_disp /= i+1
            val_total_A_loss_disp.append(val_total_A_disp)
            
            val_total_S_disp /= i+1
            val_total_S_loss_disp.append(val_total_S_disp)
            torch.save(val_total_loss,folderInitSave+'/val_total_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
            torch.save(val_total_A_loss_disp,folderInitSave+'/val_total_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
            torch.save(val_total_S_loss_disp,folderInitSave+'/val_total_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
            
            
            test_total = 0
            test_total_A_disp = 0
            test_total_S_disp = 0
            for i, (X,A,S) in enumerate(test_loader):
                
                if optNN == 1:
                    S_pred_pretrained_init = S_init_test
                    A_pred_pretrained_init = A_init_test
                    S_pred_tab, A_pred_tab = model(X,H_init=S_pred_pretrained_init,W_init=A_pred_pretrained_init)
                elif optNN == 0:
                    S_pred_tab, A_pred_tab = model(X,H_init=S_init_test,W_init=A_init_test)
                
                   
                test_loss,test_loss_A,test_loss_S = criterion(A,A_pred_tab[-1],S,S_pred_tab[-1])
                
                test_total += test_loss.item()
                test_total_A_disp += test_loss_A.item()
                test_total_S_disp += test_loss_S.item()

            test_total /= i+1
            test_total_loss.append(test_total)
            
            test_total_A_disp /= i+1
            test_total_A_loss_disp.append(test_total_A_disp)
            
            test_total_S_disp /= i+1
            test_total_S_loss_disp.append(test_total_S_disp)
            
            torch.save(test_total_loss,folderInitSave+'/test_total_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
            torch.save(test_total_A_loss_disp,folderInitSave+'/test_total_A_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
            torch.save(test_total_S_loss_disp,folderInitSave+'/test_total_S_loss_enCours_debut%s_mod%s.pth'%(np.mod(epoch,3),nbEpochsPretrained))
        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
        if epoch % 5 == 0:
            print()

        # print("epoch:{} | training loss:{:.5f} | validation loss:{:.5f} ".format(epoch, train_total,val_total))
        print("epoch:{} | training loss:{:.5f} ".format(epoch, train_total))
        
        # For SLURM restarts
        save_checkpoint({'epoch': epoch + 1,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict()},filename=folderInitSave + '/checkpoint.pth')
        # End SLURM restarts
		
	
    return train_total_loss, val_total_loss,model,test_total_loss,train_total_A_loss_disp,train_total_S_loss_disp,val_total_A_loss_disp,val_total_S_loss_disp,test_total_A_loss_disp,test_total_S_loss_disp
