import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import matplotlib.pyplot as plt

import src.models.transformer as transformer
import src.utils.extractor as extractor
import src.utils.utils as utils

class HSUModel():
    def __init__(self):
        pass

    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        raise NotImplementedError(f"Loss function not defined")
    
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

class weightConstraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.clamp_(min=0)

def init_decoder_weights(model, Y, c, kernel=None, is_unmixer=False, use_sivm=False):
    """
    Initializes the model's decoder weights with VCA extracted endmembers
    input Y must be of shape (B, N) or (B, H, W) -> no batch
    """
    if use_sivm:
        init_em = extractor.SiVM(Y, c)
    else:
        init_em = extractor.VCA(Y, c)
        
    model_dict = model.state_dict()
    
    if kernel is not None:
        model_dict['decoder.weight'][:, :, kernel//2, kernel//2] = init_em
    else:
        if is_unmixer:
            model_dict["decoder.decoder.weight"][:,:,0,0] = init_em
        else:
            model_dict["decoder.0.weight"][:,:,0,0] = init_em
        
    model.load_state_dict(model_dict)
    return model

"""
Autoencoders
"""

class CNNAEU(nn.Module, HSUModel):
    """
    CNNAEU implementation from the HySUPP repo
    
    Args:
        B (int): the number of spectral bands
        c (int): the number of endmembers
    """
    def __init__(self, B, c, scale=3.0, dev="cpu"):
        super().__init__()
        self.B = B
        self.c = c

        self.device = dev

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }
        
        self.init_architecture()

        self.scale = scale

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

    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        sad = utils.SADLoss()
        return sad(Y_gt, Y_hat)
    
    def forward(self, x):
        # Input shape (batch, B, N)
        # Output shapes Y: (batch, B, N)
        # E: (batch, B, c)
        # A: (batch, c, N)
        
        if x.dim() < 3:
            x = x.unsqueeze(0) # Add a batch dimension for inference
        
        batch, B, N = x.shape
        x = utils.oneD_to_2d(x)
        
        code = self.encoder(x)
        
        abund = F.softmax(code * self.scale, dim=1)
        a_hat = abund.reshape(batch, self.c, N)
        
        x_hat = self.decoder(abund)
        x_hat = x_hat.reshape(batch, B, N)
        
        e_hat = self.decoder.weight.detach().mean((2, 3))
        e_hat = e_hat.reshape(batch, self.B, self.c)
        
        return e_hat, a_hat, x_hat

class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        try:
            input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                              inp.view(-1, self.num_bands, 1)))
            target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                               target.view(-1, self.num_bands, 1)))

            summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
            angle = torch.acos(summation / (input_norm * target_norm))

        except ValueError:
            return 0.0

        return angle

class DeepTrans(nn.Module, HSUModel):
    """
    Args:
        B (int): the number of spectral bands
        c (int): the number of endmembers
        im_size (int): the height (or width) of the image (expects square image)
        patch_size (int, optional): how much to split the input image (default: 5)
        embed_dim (int, optional): the dimension of the features extracted 
    """
    def __init__(self, B, c, im_size, patch_size=5, embed_dim=24):
        super(DeepTrans, self).__init__()
        self.B, self.c, self.im_size, self.embed_dim, self.patch_size = B, c, im_size, embed_dim, patch_size
        self.encoder = nn.Sequential(
            nn.Conv2d(B, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (embed_dim*c)//patch_size**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((embed_dim*c)//patch_size**2, momentum=0.5),
        )

        self.vtrans = transformer.ViT(image_size=im_size, patch_size=patch_size, embed_dim=(embed_dim*c), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(embed_dim, im_size ** 2),
        )
        
        self.smooth = nn.Sequential(
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

    @staticmethod
    def loss(E_gt=None, E_hat=None, A_gt=None, A_hat=None, Y_gt=None, Y_hat=None, alpha=4e3, beta=5e-2):
        # mse = nn.MSELoss(reduction="sum")
        # sad = utils.SADLoss()

        # loss_re = alpha * mse(Y_gt, Y_hat)
        # loss_sad = beta * sad(Y_gt, Y_hat)
        # return loss_re + loss_sad
        B = Y_gt.shape[1]
        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = SAD(B)
        loss_re = alpha * loss_func(Y_hat, Y_gt)
        loss_sad = loss_func2(Y_hat.view(1, B, -1).transpose(1, 2),
                                Y_gt.view(1, B, -1).transpose(1, 2))
        loss_sad = beta * torch.sum(loss_sad).float()

        total_loss = loss_re + loss_sad
        return total_loss

    def forward(self, x):
        # Input shape (batch, B, N*) with N*=H*²
        # H* the highest multiple of patch that can fit in H
        # Output shapes Y: (batch, B, N)
        # E: (batch, B, c)
        # A: (batch, c, N)
        
        if x.dim() < 3:
            x = x.unsqueeze(0) # Add a batch dimension for inference

        abu_est = self.encoder(x)
        abu_est = abu_est.reshape(1, self.c, self.im_size, self.im_size)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.c, -1)
        abu_est = self.upscale(cls_emb).view(1, self.c, self.im_size, self.im_size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        
        e_est = self.decoder[0].weight.detach()[:,:,0,0]
        e_est = e_est.reshape(1, self.B, self.c)
        
        abu_est = abu_est.reshape(1, self.c, self.im_size**2)
        re_result = re_result.reshape(1, self.B, self.im_size**2)
        
        return e_est, abu_est, re_result

class DeepTransDOFA(nn.Module, HSUModel):
    """
    Args:
        B (int): the number of spectral bands
        c (int): the number of endmembers
        im_size (int): the height (or width) of the image (expects square image)
        patch_size (int, optional): how much to split the input image (default: 5)
        embed_dim (int, optional): the dimension of the features extracted 
    """
    def __init__(self, B, c, im_size, patch_size=5, embed_dim=24, use_cls=False):
        super(DeepTransDOFA, self).__init__()
        self.B, self.c, self.im_size, self.embed_dim = B, c, im_size, embed_dim
        self.patch_size, self.use_cls = patch_size, use_cls

        if self.use_cls:
            self.encoder = nn.Sequential(
                nn.Linear(int(768/c), self.im_size**2)
            )
        else:
            self.upsample = nn.Linear(14*14, im_size ** 2)
            self.encoder = nn.Sequential(
                nn.Conv2d(768, c, kernel_size=1),
                nn.LeakyReLU(0.02),
                nn.BatchNorm2d(c),
                nn.Dropout(0.2)
            )

        self.vtrans = transformer.ViT(image_size=im_size, patch_size=patch_size, embed_dim=(embed_dim*c), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(embed_dim, im_size ** 2),
        )
        
        self.smooth = nn.Sequential(
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

    @staticmethod
    def loss(E_gt=None, E_hat=None, A_gt=None, A_hat=None, Y_gt=None, Y_hat=None, alpha=4e3, beta=5e-2):

        B = Y_gt.shape[1]
        loss_func = nn.MSELoss(reduction='mean')
        loss_func2 = SAD(B)
        loss_re = alpha * loss_func(Y_hat, Y_gt)
        loss_sad = loss_func2(Y_hat.view(1, B, -1).transpose(1, 2),
                                Y_gt.view(1, B, -1).transpose(1, 2))
        loss_sad = beta * torch.sum(loss_sad).float()

        total_loss = loss_re + loss_sad
        return total_loss

    def forward(self, features):
        # Input shape (1, D) 
        # Output shapes Y: (batch, B, N)
        # E: (batch, B, c)
        # A: (batch, c, N)
        if self.use_cls:
            features_2d = features.reshape(self.c, int(768/self.c))
            abu_est = self.encoder(features_2d)
        else:
            features_up = self.upsample(features).reshape(1, 768, 224, 224)
            abu_est = self.encoder(features_up)

        abu_est = abu_est.reshape(1, self.c, self.im_size, self.im_size)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.c, -1)
        abu_est = self.upscale(cls_emb).view(1, self.c, self.im_size, self.im_size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        
        e_est = self.decoder[0].weight.detach()[:,:,0,0]
        e_est = e_est.reshape(1, self.B, self.c)
        
        abu_est = abu_est.reshape(1, self.c, self.im_size**2)
        re_result = re_result.reshape(1, self.B, self.im_size**2)
        
        return e_est, abu_est, re_result

class UnDIP(nn.Module, HSUModel):
    def __init__(self, B, c, kernel_size=3):
        super().__init__()
        self.B = B
        self.c = c
        
        self.kernel_sizes = [kernel_size] * 3 + [1] * 3
        self.strides = [2, 1, 1, 1, 1, 1]
        self.padding = [(k - 1) // 2 for k in self.kernel_sizes]

        self.lrelu_params = {
            "negative_slope": 0.1,
            "inplace": True,
        }
        
        self.init_architecture(seed=0)

    def init_architecture(self,seed):
        # Set random seed
        torch.manual_seed(seed)
        # MiSiCNet-like architecture
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[0]),
            nn.Conv2d(self.B, 256, self.kernel_sizes[0], stride=self.strides[0]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[1]),
            nn.Conv2d(256, 256, self.kernel_sizes[1], stride=self.strides[1]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.layerskip = nn.Sequential(
            nn.ReflectionPad2d(self.padding[-1]),
            nn.Conv2d(self.B, 4, self.kernel_sizes[-1], stride=self.strides[-1]),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer3 = nn.Sequential(
            nn.BatchNorm2d(260),
            nn.ReflectionPad2d(self.padding[2]),
            nn.Conv2d(260, 256, self.kernel_sizes[2], stride=self.strides[2]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer4 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[3]),
            nn.Conv2d(256, 256, self.kernel_sizes[3], stride=self.strides[3]),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(**self.lrelu_params),
        )

        self.layer5 = nn.Sequential(
            nn.ReflectionPad2d(self.padding[4]),
            nn.Conv2d(256, self.c, self.kernel_sizes[4], stride=self.strides[4]),
        )

        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def custom_cat(x1, xskip):
        inputs = [x1, xskip]
        inputs_shape2 = [x.shape[2] for x in inputs]
        inputs_shape3 = [x.shape[3] for x in inputs]
        if np.all(np.array(inputs_shape2) == min(inputs_shape2)) and np.all(
            np.array(inputs_shape3) == min(inputs_shape3)
        ):
            inputs_ = inputs
        else:

            inputs_ = []

            target_shape2 = min(inputs_shape2)
            target_shape3 = min(inputs_shape3)

            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[
                        :,
                        :,
                        diff2 : diff2 + target_shape2,
                        diff3 : diff3 + target_shape3,
                    ]
                )

        return torch.cat(inputs_, dim=1)

    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        loss = F.mse_loss(Y_gt, Y_hat)
        return loss

    def forward(self, x):
        # Input shape (batch, B, N)
        # Output shapes Y: (batch, B, N)
        # E: (batch, B, c)
        # A: (batch, c, N)

        if x.dim() < 3:
            x = x.unsqueeze(0) # Add a batch dimension for inference

        batch, B, N = x.shape
        x = utils.oneD_to_2d(x)
        
        x1 = self.upsample(self.layer2(self.layer1(x)))
        xskip = self.layerskip(x)
        xcat = self.custom_cat(x1, xskip)
        a_hat = self.softmax(self.layer5(self.layer4(self.layer3(xcat))))
        a_hat = a_hat.reshape(batch, a_hat.shape[1], N)
        
        x_flat = x.reshape(batch, B, N)

        e_hat = []
        for b in range(batch):
            e_hat.append(extractor.SiVM(x_flat[b], self.c))
            
        e_hat = torch.stack(e_hat, dim=0)
        y_hat = e_hat @ a_hat
        
        return e_hat, a_hat, y_hat
    
"""
Unrolling
"""

class MLP(nn.Module):
    """
    Simple MLP architecture for Ae
    
    Args:
        input_size (list): the shape of Endmember matrix E (default: [65, 4])
    """
    def __init__(self, input_size=[65,4]):
        super().__init__()
        self.input_size = input_size
        self.a1 = nn.Linear(input_size[0]*input_size[1],130,dtype=torch.float)
        self.a2 = nn.Linear(130,75,dtype=torch.float)
        self.a3 = nn.Linear(75,input_size[0]*input_size[1],dtype=torch.float)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        y = x.reshape((x.shape[0],self.input_size[0]*self.input_size[1]))
        
        out = self.a1(y)
        out = self.relu(out)
        
        out = self.a2(out)
        out = self.relu(out)
    
        out = self.a3(out)
      
        output = out.reshape((x.shape[0],self.input_size[0],self.input_size[1]))
        return output
    
class CNN2D(nn.Module):
    """
    Simple 2D CNN architecture for Aa
    
    Args:
        input_size (list): the shape of matrix Abundance A (default: [4, 346, 346])
        conv_size (int): the size of the convolution kernels (default: 3)
    """
    def __init__(self, input_size=[4,346,346],conv_size=5):
        super().__init__()
        self.input_size = input_size
        
        # On garde un nombre de canaux egaux au nombre de sources
        self.conv1 = nn.Conv2d(input_size[0], 32, conv_size, padding='same',dtype=torch.float)
        self.conv2 = nn.Conv2d(32, 32, conv_size, padding='same',dtype=torch.float)
        self.conv3 = nn.Conv2d(32, 16, conv_size, padding='same',dtype=torch.float)
        self.conv4 = nn.Conv2d(16, 8, conv_size, padding='same',dtype=torch.float)
        self.conv5 = nn.Conv2d(8, input_size[0], conv_size, padding='same',dtype=torch.float)
        self.relu = nn.ReLU()
        
    def forward(self,x):

        out = self.conv1(x)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.relu(out)
    
        out = self.conv3(out)
        out = self.relu(out)
        
        out = self.conv4(out)
        out = self.relu(out)
        
        out = self.conv5(out)
        out = self.relu(out)
        
        return out
    
class NALMU(nn.Module, HSUModel):
    """
    Defines the NALMU model: Ee unroll the MU algorithm to update the endmembers E and abundances A matrices
    
    Args:
        T (int, optional): Number of layers in the unrolled neural network (default: 10)
        B (int, optional): Number of observations (eg Wavelengt bands) (default: 64)
        c (int, optional): Number of sources (eg endmembers) (default: 4)
        N (int, optional): Number of samples (eg pixels) (default: 10000)
        shared (bool, optional): Whether to share the weights across unrolled layers or not (default: False)
    """
    def __init__(self, T=10, B=64, c=4, N=10000, shared=False):
        super(NALMU, self).__init__()
        
        self.T = T 
        self.B = B 
        self.c = c 
        self.N = N
        self.shared = shared
        
        self.Ae = torch.zeros((B,c)).to(torch.float32)
        # self.Aa = torch.zeros((c,N)).to(torch.float32)
            
        if not self.shared:
            self.Ae = self.Ae.repeat(self.T,1,1)
            # self.Aa = self.Aa.repeat(self.T,1,1)
            
        self.Ae = nn.Parameter(self.Ae,requires_grad = True)
        # self.Aa = nn.Parameter(self.Aa,requires_grad = True)

    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        num_E = E_hat.shape[2]

        if E_hat.dim() != 3:
            E_hat = E_hat.unsqueeze(0)

        sad = utils.SADLoss()
        mse = nn.MSELoss(reduction='sum')

        E_ordered, E_ordered_norm, A_ordered, indices = utils.order_endmembers(E_gt, E_hat, A_hat)
        E_ordered = E_ordered[0]
        E_gt = E_gt[0]
        A_ordered = A_ordered[0]
        
        train_A = mse(A_gt,A_ordered)/(torch.norm(A_gt)**2)
        train_E = sad(E_gt,E_ordered)

        return train_A + train_E
    
    def forward(self, X, E_init=None, A_init=None, epoch=-1):
        
        if X.dim() < 3:
            X = X.unsqueeze(0) # Add a batch dimension for inference
        
        b_size = X.shape[0]
        
        E_pred_tab = []
        A_pred_tab = []
        
        # Initialize A and E
        if A_init == None:
            A_init = torch.ones(b_size, self.c, self.N)
        if E_init == None:
            E_init = torch.ones(b_size, self.B, self.c)
        
        # Initialize E and A
        E_pred = E_init.clone()
        A_pred = A_init.clone()

        for t in range(self.T):
            if self.shared:
                A_pred = A_pred * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            else: # If parameters not shared
                A_pred = A_pred * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            
            A_pred = A_pred.clip(min=1e-7,max=1)
            
            if self.shared:
                E_pred = E_pred*torch.exp(self.Ae.repeat(b_size,1,1)) * torch.bmm(X,torch.transpose(A_pred,1,2))/(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2)))
            else:
                E_pred = E_pred*torch.exp(self.Ae[t].repeat(b_size,1,1)) * torch.bmm(X,torch.transpose(A_pred,1,2))/(torch.bmm(torch.bmm(E_pred,A_pred),torch.transpose(A_pred,1,2)))
            
            E_pred = E_pred.clip(min=1e-7,max=1e4)
                
            A_pred_tab.append(A_pred)
            E_pred_tab.append(E_pred)
            
        E_est = E_pred_tab[-1]
        A_est = A_pred_tab[-1]

        X_reconstruct = E_est @ A_est

        return E_est, A_est, X_reconstruct
    
class RALMU(nn.Module, HSUModel):
    """
    Defines the RALMU model: Ee unroll 
    
    Args:
        T (int, optional): Number of layers in the unrolled neural network (default: 10)
        B (int, optional): Number of observations (eg Waveleight bands) (default: 64)
        c (int, optional): Number of sources (eg endmembers) (default: 4)
        shared (bool, optional): Whether to share the weights across unrolled layers or not (default: False)
        conv_size (int, optional): the kernel size of the 2D-CNN for Aa (default: 3)
        im_size (int, optional): The input image's height (or width), expects square images (default: 256)
    """
    def __init__(self, T=10, B=64, c=4, shared=False, conv_size=5, im_size=256):
        super(RALMU, self).__init__()
        
        self.T = T
        self.B = B
        self.c = c
        self.shared = shared
        self.size_image_A = [c, im_size, im_size]
        
        if self.shared:
            tab_mlp_E = nn.ParameterList([MLP((B,c))])
        else:
            tab_mlp_E = nn.ParameterList([])
            for _ in range(T):
                tab_mlp_E.append(MLP((B,c)))
        # On pourrait ici pre-entrainer les reseaux a predire un E appris par un premier reseau
        self.tab_mlp_E = tab_mlp_E
        
        if self.shared:
            tab_mlp_A = nn.ParameterList([CNN2D(self.size_image_A, conv_size=conv_size)])
        else:
            tab_mlp_A = nn.ParameterList([])
            for _ in range(T):
                tab_mlp_A.append(CNN2D(self.size_image_A, conv_size=conv_size))
        
        # On pourrait ici pre-entrainer les reseaux a predir un A appris par un premier reseau
        self.tab_mlp_A = tab_mlp_A

    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        # num_E = E_hat.shape[2]

        if E_hat.dim() != 3:
            E_hat = E_hat.unsqueeze(0)

        sad = utils.SADLoss()
        mse = nn.MSELoss(reduction='sum')

        E_ordered, E_ordered_norm, A_ordered, indices = utils.order_endmembers(E_gt, E_hat, A_hat)
        E_ordered = E_ordered[0]
        E_gt = E_gt[0]
        A_ordered = A_ordered[0]
        train_A = mse(A_gt,A_ordered)/(torch.norm(A_gt)**2)
        train_E = sad(E_gt,E_ordered)

        return train_A + train_E

    def forward(self, X, E_init=None, A_init=None):
        # A_initNetA : of shape (nb batchs, nb sources, nb pixel), i.e. a vectorized image
    
        E_pred_tab = []
        A_pred_tab = []
        
        if X.dim() < 3:
            X = X.unsqueeze(0) # Add a batch dimension for inference
            
        b_size = X.shape[0]
        
        # Initialize A and E
        if A_init == None:
            A_init = torch.ones(b_size, self.c, self.size_image_A[1]**2)
        if E_init == None:
            E_init = torch.ones(b_size, self.B, self.c)
        
        A_pred = A_init.clone()
        E_pred = E_init.clone()
        
        # A_init_im = torch.reshape(A_init, (b_size, self.size_image_A[0], self.size_image_A[1], self.size_image_A[2]))
        
        soft = torch.nn.Softplus()
        
        for t in range(self.T):
            
            #------------- Partie sur A --------------
            if self.shared:
                Aa = self.tab_mlp_A[0](A_pred.reshape((b_size, self.size_image_A[0], self.size_image_A[1], self.size_image_A[2])))
            else:
                Aa = self.tab_mlp_A[t](A_pred.reshape((b_size, self.size_image_A[0], self.size_image_A[1], self.size_image_A[2])))
                    
            Aa = torch.reshape(Aa, (b_size, self.size_image_A[0], self.size_image_A[1]*self.size_image_A[2]))

            Aa = soft(Aa)# For nonnegativity

            A_pred = A_pred * Aa * torch.bmm(torch.transpose(E_pred,1,2),X)/(torch.bmm(torch.transpose(E_pred,1,2),torch.bmm(E_pred,A_pred)))
            A_pred = A_pred.clip(min=1e-7,max=1e4)

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

            A_pred_tab.append(A_pred)
            E_pred_tab.append(E_pred)
            
        E_est = E_pred_tab[-1]
        A_est = A_pred_tab[-1]
        X_reconstruct = E_est @ A_est

        return E_est, A_est, X_reconstruct
    

"""
Other methods
"""

class MU(nn.Module):
    def __init__(self, N_iter=int(1e5)):
        super(MU, self).__init__()
        self.N_iter = N_iter
    
    def init_A(self, batch, c, N):
        A_init = torch.ones(batch, c, N)
        return A_init
    
    def init_E(self, batch, B, c):
        E_init = torch.ones(batch, B, c)
        return E_init
    
    def forward(self, x, c):
        if x.dim() < 3:
            x = x.unsqueeze(0)
            
        batch, B, N = x.shape
        
        A_hat = self.init_A(batch, c, N)
        E_hat = self.init_E(batch, B, c)
        
        E_hat_tab = []
        A_hat_tab = []
        Y_hat_tab = []
        
        for n in range(self.N_iter):
            A_hat = A_hat * torch.bmm(torch.transpose(E_hat, 1, 2), x)/ (torch.bmm(torch.transpose(E_hat, 1, 2), torch.bmm(E_hat, A_hat)))
            A_hat = A_hat.clip(min=1e-7, max=1)
            
            E_hat = E_hat * torch.bmm(x, torch.transpose(A_hat, 1, 2))/ (torch.bmm(torch.bmm(E_hat, A_hat), torch.transpose(A_hat, 1, 2)))
            E_hat = E_hat.clip(min=1e-7,max=1e4)
                
            A_hat_tab.append(A_hat)
            E_hat_tab.append(E_hat)
            Y_hat_tab.append(E_hat @ A_hat)
        
        A_hat = A_hat_tab[-1]
        E_hat = E_hat_tab[-1]
        
        Y_hat = Y_hat_tab[-1]
        
        return E_hat, A_hat, Y_hat, Y_hat_tab