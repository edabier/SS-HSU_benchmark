import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import transformer
from sklearn.feature_extraction.image import extract_patches_2d
import tqdm

class HSUModel():
    def __init__(self):
        pass
    
    def forward(self, x):
        """
        Args:
            x: input HSI to unmix (shape (B,N))
        Returns:
            E: the endmember matrix (shape (B, c))
            A: the abundance matrix (shape (c, N))
            x_hat: the reconstructed HSI (shape (B,N))
        """
        raise NotImplementedError(f"Forward method not implemented for {self}")

"""
Autoencoders
"""

class CNNAE_linear(nn.Module, HSUModel):
    """
    Adaptation of the CNNAEU implementation from the HySUPP repo
    """
    def __init__(self, scale=3.0, epochs=200, lr=0.001, batch_size=1, patch_size=40):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }

        self.scale = scale
        # self.epochs = epochs
        # self.lr = lr
        # self.batch_size = batch_size
        # self.patch_size = patch_size

    def init_architecture(self, seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.encoder = nn.Sequential(
            nn.Conv2d(self.B, 48, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.c, kernel_size=1, bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(self.c),
            nn.Dropout2d(p=0.2),
        )

        self.decoder = nn.Linear(in_features=self.H*self.W*self.c, out_features=self.H*self.W*self.B)

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        e_est = self.decoder.weight
        return e_est, abund, x_hat

    # @staticmethod
    # def loss(target, output):
    #     assert target.shape == output.shape

    #     dot_product = (target * output).sum(dim=1)
    #     target_norm = target.norm(dim=1)
    #     output_norm = output.norm(dim=1)
    #     sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
    #     return sad_score.mean()

    # def unmix(self, Y, c, H, W, seed=0):
    #     tic = time.time()

    #     B, N = Y.shape
    #     # Hyperparameters
    #     self.B = B  # number of spectral bands
    #     self.c = c  # number of endmembers
    #     self.H = H  # number of lines
    #     self.W = W  # number of samples per line

    #     self.num_patches = int(250 * self.H * self.W * self.B / (307 * 307 * 162))

    #     self.init_architecture(seed=seed)

    #     num_channels, h, w = self.B, self.H, self.W

    #     Y_numpy = Y.reshape((num_channels, h, w)).transpose((1, 2, 0))

    #     input_patches = extract_patches_2d(
    #         Y_numpy,
    #         max_patches=self.num_patches,
    #         patch_size=(self.patch_size, self.patch_size),
    #     )
    #     input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

    #     # Send model to GPU
    #     self = self.to(self.device)
    #     optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

    #     # Dataloader
    #     dataloader = torch.utils.data.DataLoader(
    #         input_patches,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    #     progress = tqdm(range(self.epochs))
    #     for ee in progress:

    #         running_loss = 0
    #         for ii, batch in enumerate(dataloader):
    #             batch = batch.to(self.device)
    #             optimizer.zero_grad()

    #             _, outputs = self(batch)

    #             # Reshape data
    #             loss = self.loss(batch, outputs)
    #             running_loss += loss.item()

    #             loss.backEard()
    #             optimizer.step()

    #         progress.set_postfix_str(f"loss={running_loss:.3e}")

    #     # Get final abundances and endmembers
    #     self.eval()

    #     Y_eval = torch.Tensor(Y.reshape((1, num_channels, h, w))).to(self.device)

    #     abund, _ = self(Y_eval)

    #     Ahat = abund.detach().cpu().numpy().reshape(self.c, self.H * self.W)
    #     Ehat = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

    #     self.time = time.time() - tic

    #     return Ehat, Ahat

class CNNAEU(nn.Module, HSUModel):
    """
    CNNAEU implementation from the HySUPP repo
    """
    def __init__(self, scale=3.0, epochs=200, lr=0.001, batch_size=1, patch_size=40):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }

        self.scale = scale
        # self.epochs = epochs
        # self.lr = lr
        # self.batch_size = batch_size
        # self.patch_size = patch_size

    def init_architecture(self, seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.encoder = nn.Sequential(
            nn.Conv2d(self.B, 48, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.c, kernel_size=1, bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(self.c),
            nn.Dropout2d(p=0.2),
        )

        self.decoder = nn.Conv2d(self.c, self.B, kernel_size=11, padding=5, padding_mode="reflect", bias=False)

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        e_est = ?
        return e_est, abund, x_hat

    # @staticmethod
    # def loss(target, output):
    #     assert target.shape == output.shape

    #     dot_product = (target * output).sum(dim=1)
    #     target_norm = target.norm(dim=1)
    #     output_norm = output.norm(dim=1)
    #     sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
    #     return sad_score.mean()

    # def unmix(self, Y, c, H, W, seed=None):
    #     tic = time.time()

    #     B, N = Y.shape
    #     # Hyperparameters
    #     self.B = B  # number of spectral bands
    #     self.c = c  # number of endmembers
    #     self.H = H  # number of lines
    #     self.W = W  # number of samples per line

    #     self.num_patches = int(250 * self.H * self.W * self.B / (307 * 307 * 162))

    #     self.init_architecture(seed=seed)

    #     num_channels, h, w = self.B, self.H, self.W

    #     Y_numpy = Y.reshape((num_channels, h, w)).transpose((1, 2, 0))

    #     input_patches = extract_patches_2d(
    #         Y_numpy,
    #         max_patches=self.num_patches,
    #         patch_size=(self.patch_size, self.patch_size),
    #     )
    #     input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

    #     # Send model to GPU
    #     self = self.to(self.device)
    #     optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

    #     # Dataloader
    #     dataloader = torch.utils.data.DataLoader(
    #         input_patches,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #     )

    #     progress = tqdm(range(self.epochs))
    #     for ee in progress:

    #         running_loss = 0
    #         for ii, batch in enumerate(dataloader):
    #             batch = batch.to(self.device)
    #             optimizer.zero_grad()

    #             _, outputs = self(batch)

    #             # Reshape data
    #             loss = self.loss(batch, outputs)
    #             running_loss += loss.item()

    #             loss.backEard()
    #             optimizer.step()

    #         progress.set_postfix_str(f"loss={running_loss:.3e}")

    #     # Get final abundances and endmembers
    #     self.eval()

    #     Y_eval = torch.Tensor(Y.reshape((1, num_channels, h, w))).to(self.device)

    #     abund, _ = self(Y_eval)

    #     Ahat = abund.detach().cpu().numpy().reshape(self.c, self.H * self.W)
    #     Ehat = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

    #     self.time = time.time() - tic

    #     return Ehat, Ahat

class Transformer_AE(nn.Module, HSUModel):
    """
    Args:
        c (int): the number of endmembers
        B (int): the number of spectral bands
    """
    def __init__(self, c, B, size, patch, dim):
        super(Transformer_AE, self).__init__()
        self.c, self.B, self.size, self.dim = c, B, size, dim
        self.encoder = nn.Sequential(
            nn.Conv2d(B, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*c)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*c)//patch**2, momentum=0.5),
        )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*c), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        
        self.smootA = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(c, B, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.vieE(1, self.c, -1)
        abu_est = self.upscale(cls_emb).vieE(1, self.c, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        e_est = ?
        return e_est, abu_est, re_result


"""
Unrolling
"""

class MLP(nn.Module):
    """
    Simple MLP architecture for Ae
    """
    def __init__(self, m, n):
        super().__init__()
        
        self.fc = nn.Linear(m,n)
    
    def forward(self, x):
        out = self.fc(x)
        return out
    
class CNN2D(nn.Module):
    """
    Simple 2D CNN architecture for Aa
    """
    def __init__(self, image_size, conv_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=image_size, out_channels=image_size, kernel_size=conv_size)
    
    def forward(self, x):
        out = self.conv1(x)
        return out
    
class NALMU(nn.Module, HSUModel):
    """
    Defines the NALMU model: Ee unroll the MU algorithm to update the endmembers E and abundances A matrices
    
    Args:
        T (int, optional): Number of layers in the unrolled neural network (default: 10)
        b (int, optional): Number of observations (eg Wavelengt bands) (default: 64)
        c (int, optional): Number of sources (eg endmembers) (default: 4)
        n (int, optional): Number of samples (eg pixels) (default: 10000)
        shared (bool, optional): Whether to share the weights across unrolled layers or not (default: False)
    """
    def __init__(self, T=10, b=64, c=4, n=10000, shared=False):
        super(NALMU, self).__init__()
        
        self.T = T 
        self.b = b 
        self.c = c 
        self.n = n
        self.shared = shared
        
        # A cause de la nonnegativite, on va utiliser exp(Ae) plutot que Ae => pour que exp(Ae) ne comprenne que des 1, il faut que Ae n'ait que des zeros
        self.Ae = torch.zeros((b,c)).to(torch.float32) # Ae doit etre le meme pour tous les minibatchs, on ne prend qu'un seul E pour l'initialisation (arbitrairement, le premier du mini-batch) et on fait un repeat dans les iterations du LMU
        
        # A cause de la nonnegativite, on va utiliser exp(Aa) plutot que Aa => pour que exp(Aa) ne comprenne que des 1, il faut que Aa n'ait que des zeros
        self.Aa = torch.zeros((c,n)).to(torch.float32) # Aa doit etre le meme pour tous les minibatchs, on ne prend qu'un seul A pour l'initialisation (arbitrairement, le premier du mini-batch) et on fait un repeat dans les iterations du LMU
            
        #Ae est de taille [m,n]
        if not self.shared:
            self.Ae = self.Ae.repeat(self.T,1,1) # De taille [T,m,n]
            self.Aa = self.Aa.repeat(self.T,1,1) # De taille [T,n,t].
            
        self.Ae = nn.Parameter(self.Ae,requires_grad = True)
        self.Aa = nn.Parameter(self.Aa,requires_grad = True)
            
    def forward(self, X, E_init=None, A_init=None, epoch=-1):
        # Remarque : les tailles de E sont fixees des l'initialisation du reseau et celle de A l'est Mais on peut se servir de E_init et A_init pour initialiser le reseau, par exemple avec un VCA.
        b_size = X.shape[0]
        
        E_pred_tab = []
        A_pred_tab = []
        # Initialize E and A
        E_pred = E_init
        A_pred = A_init

        for t in range(self.T):
            if hasattr(self, 'shared') and self.shared:
                A_pred = A_pred * torch.exp(self.Aa.repeat(b_size,1,1)) * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            else: # If parameters not shared
                A_pred = A_pred * torch.exp(self.Aa[t].repeat(b_size,1,1)) * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            
            A_pred = A_pred.clip(min=1e-7,max=1)
            
            if hasattr(self, 'shared') and self.shared:
                E_pred = E_pred*torch.exp(self.Ae.repeat(b_size,1,1)) * torch.bmm(X,torch.transpose(A_pred,1,2))/(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2)))
            else:
                E_pred = E_pred*torch.exp(self.Ae[t].repeat(b_size,1,1)) * torch.bmm(X,torch.transpose(A_pred,1,2))/(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2)))
            
            E_pred = E_pred.clip(min=1e-7,max=1e4)
                
            A_pred_tab.append(A_pred)
            E_pred_tab.append(E_pred)
            
        E_est = torch.stack(E_pred_tab, dim=)
        A_est = torch.stack(A_pred_tab, dim=)
        X_reconstruct = E_est @ A_est

        return E_est, A_est, X_reconstruct
    
class RALMU(nn.Module, HSUModel):
    """
    Defines the RALMU model: Ee unroll 
    
    Args:
        T (int, optional): Number of layers in the unrolled neural network (default: 10)
        b (int, optional): Number of observations (eg Waveleight bands) (default: 64)
        c (int, optional): Number of sources (eg endmembers) (default: 4)
        shared (bool, optional): Whether to share the weights across unrolled layers or not (default: False)
        conv_size (int, optional): the kernel size of the 2D-CNN for Aa (default: 3)
        size_A_image (list, optional): The input image size (default: [4,256,256])
    """
    def __init__(self, T=10, b=64, c=4, shared=False, conv_size=3, size_A_image=[4,256,256]):
        super(RALMU, self).__init__()
        
        self.T = T
        self.shared = shared
        self.size_A_image = size_A_image
        
        if self.shared:
            tab_mlp_E = nn.ParameterList([MLP((b,c))])
        else:
            tab_mlp_E = nn.ParameterList([])
            for ii in range(T):
                tab_mlp_E.append(MLP((b,c)))
        # On pourrait ici pre-entrainer les reseaux a predire un E appris par un premier reseau
        self.tab_mlp_E = tab_mlp_E
        
        if self.shared:
            tab_mlp_A = nn.ParameterList([CNN2D(self.size_A_image, conv_size=conv_size)])
        else:
            tab_mlp_A = nn.ParameterList([])
            for ii in range(T):
                tab_mlp_A.append(CNN2D(self.size_A_image, conv_size=conv_size))
        
        # On pourrait ici pre-entrainer les reseaux a predir un A appris par un premier reseau
        self.tab_mlp_A = tab_mlp_A

        
    def forward(self, X, E_init=None, A_init=None, indMbDisp=-1):
        # A_initNetA : of shape (nb batchs, nb sources, nb pixel), i.e. a vectorized image
        A_init_im = torch.reshape(A_init, (X.size()[0],self.size_A_image[0],self.size_A_image[1],self.size_A_image[2]))
        
        E_pred_tab = []
        A_pred_tab = []
        
        # Initialize A
        A_pred = A_init.clone()
        
        # Initialize E
        E_pred = E_init.clone()
        
        soft = torch.nn.Softplus()
        
        for t in range(self.T):
            
            #------------- Partie sur A --------------
            if self.shared:
                Aa = self.tab_mlp_A[0](A_pred.reshape((X.size()[0],self.size_A_image[0],self.size_A_image[1],self.size_A_image[2])))
            else:
                Aa = self.tab_mlp_A[t](A_pred.reshape((X.size()[0],self.size_A_image[0],self.size_A_image[1],self.size_A_image[2])))
                    
            Aa = torch.reshape(Aa, (X.size()[0],self.size_A_image[0],self.size_A_image[1]*self.size_A_image[2]))

            Aa = soft(Aa)# For nonnegativity

            A_pred = A_pred * Aa * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            A_pred = A_pred.clip(min=1e-7,max=1e4) # CK : tenter de mettre max = 1 ne semble pas vraiment amÃ©liorer
            if indMbDisp >= 0:
                print('A_pred: layer %s,minibatcA %s, minimum %s:'%(t,indMbDisp,A_pred.min()))
                print('A_pred: layer %s,minibatcA %s, maximum %s:'%(t,indMbDisp,A_pred.max()))
                print('grad A den min %s'%(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred))).min())
                print('grad A  den max %s'%(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred))).max())
                
            #------------- Partie sur E ---------------
            if t == 0: 
                if self.shared:
                    Ae = self.tab_mlp_E[0](E_init) # Contrairement au cas ou E est fixe, il n'y a pas ici de repeat car on veut un E par A
                else:
                    Ae = self.tab_mlp_E[t](E_init)
            else:
                if self.shared:
                    Ae = self.tab_mlp_E[0](E_pred) # Contrairement au cas ou E est fixe, il n'y a pas ici de repeat car on veut un E par A
                else:
                    Ae = self.tab_mlp_E[t](E_pred)                
                
            Ae = soft(Ae) # For enforcing nonnegativity
                
            E_pred = E_pred*Ae * torch.bmm(X,torch.transpose(A_pred,1,2))/(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2)))
            
            E_pred = E_pred.clip(min=1e-7,max=1e4)
            if indMbDisp >= 0:
                print('E_pred: layer %s,minibatcA %s, minimum %s:'%(t,indMbDisp,E_pred.min()))
                print('E_pred: layer %s,minibatcA %s, maximum %s:'%(t,indMbDisp,E_pred.max()))
                print('grad E den min %s'%(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2))).min())
                print('grad E den max %s'%(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2))).max())

            A_pred_tab.append(A_pred)
            E_pred_tab.append(E_pred)
            
        E_est = torch.stack(E_pred_tab, dim=)
        A_est = torch.stack(A_pred_tab, dim=)
        X_reconstruct = E_est @ A_est

        return E_est, A_est, X_reconstruct