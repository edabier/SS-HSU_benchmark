import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import upsamplers
from src.models.foundation_models import Sum_to_one, Decoder
from src.utils import losses
from src.utils import utils

class UnmixingFromFeatures(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, n_features=1, upsampler="Linear", channel_selector=False):
        """
        Upsamples low res features then estimates A_hat
        
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features
            n_features (int): The size of the list of features in the case of several extracted features

        """
        super(UnmixingFromFeatures, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H
        self.n_features = n_features
        self.upsampler = upsampler

        # Upsampling features
        if upsampler == "Linear":
            self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)
        elif upsampler == "FiLM":
            self.upsample = upsamplers.FiLMUpsampler(self.n_features*D, self.n_features*D, B, alpha, H, group_channels=False)
        elif upsampler == "FiLM_grouped":
            self.upsample = upsamplers.FiLMUpsampler(self.n_features*D, self.n_features*D, B, alpha, H, group_channels=True)
        elif upsampler == "Features_fusion":
            self.upsample = upsamplers.FeaturesFusionUpsampler(self.n_features*D, B, alpha, H, group_channels=False)
        elif upsampler == "Features_fusion_grouped":
            self.upsample = upsamplers.FeaturesFusionUpsampler(self.n_features*D, B, alpha, H, group_channels=True)
        else:
            assert "Unknown upsampler, must be one of [Linear, FiLM, FiLM_grouped, Features_fusion, Features_fusion_grouped]"

        # Upsampled features to abundances
        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(self.n_features*D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        if channel_selector:
        # self.channel_selector = nn.Parameter(torch.ones(self.n_features*self.D))
            self.channel_selector = nn.Parameter(torch.ones(self.n_features))

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, features_hr=None, features_lr=None, channel_selector=None, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_tv_a=0, W_mse=0.09, W_e=0, W_feat=0, W_tv_feat=0, hypersigma=False, return_losses=False):
        sad = losses.SADLoss()
        tv = losses.TVLoss(reduction="mean")
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        if hypersigma:
            loss_mse = W_mse * losses.hypersigma_mse(Y_gt, Y_hat)
        else:
            loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""
        
        # Enforce Sum of all wavelength to be 1 for each endmember:
        # Minimize 1 - sum(E[:,i]) for i in c
        loss_norm_e = 0 #W_e * torch.norm(torch.ones(E_hat.shape[1]) - torch.sum(E_hat, dim=0))

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        # TV on abundances (sum of difference between consecutive horizontal pixels + vertical pixels)
        loss_tv_a = 0 #W_tv_a * tv(A_hat)

        """Feature regularisation"""

        if features_hr != None:
            l1 = nn.L1Loss()
            downsample = nn.AdaptiveAvgPool2d(features_lr[0].shape)
            features_down = downsample(features_hr)

            loss_features = W_feat * l1(utils.normalize(features_down), utils.normalize(features_lr))
        
        else:
            loss_features = 0

        loss = loss_sad + loss_ab + loss_tv_e + loss_tv_a + loss_mse + loss_norm_e + loss_features

        if channel_selector != None:
            loss += torch.linalg.norm(channel_selector, dim=0, ord=1)#/ len(channel_selector)

        if return_losses:
            return loss, loss_sad, loss_ab, loss_tv_e, loss_tv_a, loss_mse, loss_norm_e
        else:
            return loss

    def get_abundances(self, features, Y):

        if hasattr(self, "channem_selector"):
            if len(self.channel_selector) == self.n_features:
                weights = self.channel_selector.view(-1, 1, 1)
                features = features.view(self.n_features, self.D, self.alpha**2)
                features = features * weights
                features = features.view(-1, self.alpha**2)
            else:
                weights = self.channel_selector.view(-1, 1)
                features = features * weights

        if self.upsampler == "Linear":
            # features = features.reshape(self.n_features*self.D, self.alpha*self.alpha)
            features_up = self.upsample(features)
        else:
            features = utils.oneD_to_2d(features).unsqueeze(0)
            features_up = self.upsample(features, Y)
        features_up = features_up.view(
            1, self.n_features*self.D, self.H, self.H
        )
        features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
        A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)
        # A_hat = self.smooth(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features, Y):
        A_hat = self.get_abundances(features, Y)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class UnmixingFromFeatures2(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, n_features=1, upsampler="FiLM"):
        """
        Estimates A_hat from low res features then upsamples A_hat
        
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features
            n_features (optional : int): The size of the list of features in the case of several extracted features (default 1)

        """
        super(UnmixingFromFeatures2, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H
        self.n_features = n_features
        self.upsampler = upsampler

        # Upsampling features
        if upsampler == "Linear":
            self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)
        elif upsampler == "FiLM":
            self.upsample = upsamplers.FiLMUpsampler(c, c, B, alpha, H, group_channels=False)
        elif upsampler == "FiLM_grouped":
            self.upsample = upsamplers.FiLMUpsampler(c, c, B, alpha, H, group_channels=True)
        elif upsampler == "Features_fusion":
            self.upsample = upsamplers.FeaturesFusionUpsampler(c, B, alpha, H, group_channels=False)
        elif upsampler == "Features_fusion_grouped":
            self.upsample = upsamplers.FeaturesFusionUpsampler(c, B, alpha, H, group_channels=True)
        else:
            assert "Unknown upsampler, must be one of [Linear, FiLM, FiLM_grouped, Features_fusion, Features_fusion_grouped]"

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(self.n_features*D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        # self.channel_selector = nn.Parameter(torch.ones(self.n_features*self.D))
        self.channel_selector = nn.Parameter(torch.ones(self.n_features))

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, channel_selector=None, alpha=None, Y_hat_lr=None, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09, return_losses=False):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')

        if alpha != None:
            downsample = nn.AdaptiveAvgPool2d(alpha)
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()
        loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        """Reconstruction low resolution"""
        if Y_hat_lr != None:
            Y_lr = downsample(Y_hat)
            loss_re_down = sad(Y_lr, Y_hat_lr) + W_mse * mse(Y_lr, Y_hat_lr)
        else:
            loss_re_down = 0

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse + loss_re_down

        if channel_selector != None:
            loss += torch.linalg.norm(channel_selector, dim=0, ord=1)#/ len(channel_selector)

        if return_losses:
            return loss, loss_sad, loss_ab, loss_tv_e, loss_mse
        else:
            return loss

    def get_abundances(self, features, Y):

        if len(self.channel_selector) == self.n_features:
            weights = self.channel_selector.view(-1, 1, 1)
            features = features.view(self.n_features, self.D, self.alpha**2)
            features = features * weights
            features = features.view(-1, self.alpha**2)
        else:
            weights = self.channel_selector.view(-1, 1)
            features = features * weights
        
        features = utils.oneD_to_2d(features)
        A_hat_lr = self.abundance_estimator(features.unsqueeze(0))
        A_hat_lr = self.sum_to_one(A_hat_lr)

        if self.upsampler == "Linear":
            A_hat_lr = A_hat_lr.reshape(1, self.c, self.alpha*self.alpha)
            A_hat = self.upsample(A_hat_lr)
            A_hat = utils.oneD_to_2d(A_hat)
        else:
            A_hat = self.upsample(A_hat_lr, Y)
            
        A_hat = self.sum_to_one(A_hat)

        return A_hat, A_hat_lr
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features, Y):
        A_hat, A_hat_lr = self.get_abundances(features, Y)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat #, A_hat_lr
 
class UnmixingFromUpFeat(nn.Module):
    def __init__(self, D, B, c, H=224):
        """
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi

        """
        super(UnmixingFromUpFeat, self).__init__()
        self.D = D
        self.B = B
        self.c = c
        self.H = H

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_tv_a=0):
        sad = losses.SADLoss()
        tv = losses.TVLoss(reduction="mean")
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        # TV on abundances (sum of difference between consecutive horizontal pixels + vertical pixels)
        loss_tv_a = W_tv_a * tv(A_hat)

        loss = loss_sad + loss_ab + loss_tv_e + loss_tv_a

        return loss

    def get_abundances(self, features):
        # features = (features - features.mean())/ (1e-8 + features.std())
        A_hat = self.abundance_estimator(features)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class UnmixingFromHSI(nn.Module):
    def __init__(self, B, c, H=224):
        """
        Args:
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi

        """
        super(UnmixingFromHSI, self).__init__()
        self.B = B
        self.c = c
        self.H = H

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(B, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.6):
        sad = losses.SADLoss()
        tv = losses.TVLoss(reduction="mean")
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        return loss, loss_sad, loss_ab, loss_tv_e, loss_mse

    def get_abundances(self, Y):

        A_hat = self.abundance_estimator(Y)

        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, Y):
        A_hat = self.get_abundances(Y)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class CNNAEU_with_decoder(nn.Module):
    def __init__(self, B, c, decoder=None, E_init=None, freeze_E=True):
        super().__init__()

        self.B = B
        self.c = c
        self.encoder = nn.Sequential(
            nn.Conv2d(self.B, 48, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.c, kernel_size=1, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.c),
            nn.Dropout2d(p=0.2),
        )
        if decoder != None:
            self.decoder = decoder
        else:
            self.decoder = Decoder(self.c, self.B)
            if E_init != None:
                state_dict = self.decoder.state_dict()
                state_dict["decoder.weight"] = E_init.unsqueeze(-1).unsqueeze(-1)
                self.decoder.load_state_dict(state_dict)

        if freeze_E:
            for param in self.decoder.parameters():
                param.requires_grad = False
    
    @staticmethod
    def loss(E_gt, E_hat, A_gt, A_hat, Y_gt, Y_hat):
        sad = losses.SADLoss()
        return sad(Y_gt, Y_hat)
    
    def forward(self, x):
        # Input shape (batch, B, N)
        
        if x.dim() < 3:
            x = x.unsqueeze(0) # Add a batch dimension for inference
        
        batch, B, N = x.shape
        x = utils.oneD_to_2d(x)
        
        code = self.encoder(x)
        
        abund = F.softmax(code, dim=1)
        a_hat = abund.reshape(batch, self.c, N)
        
        x_hat = self.decoder(abund)
        x_hat = x_hat.reshape(batch, B, N)
        
        e_hat = self.decoder.get_endmembers()
        
        return e_hat, a_hat, x_hat