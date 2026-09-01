import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import upsamplers
from src.models import transformer, srvit
from src.utils import losses, utils

class HyperSLUnmixer(nn.Module):
    def __init__(self, D, alpha, H, B, c):
        super(HyperSLUnmixer, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H

        # Upsampling features
        # self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)

        self.convblock = nn.Sequential(
            nn.Conv2d(self.D,64,1,1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1,1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # h,w /2
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, self.c, 3, 1, 1),  # h,w /2
            nn.BatchNorm2d(self.c),
            nn.GELU(),
        )
        self.sum_to_one = Sum_to_one(1)
        self.decoder = Decoder(self.c, self.B, kernel_size=1)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())
        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        return loss

    def get_abundances(self, features):        
        
        # features = self.upsample(features)
        features = features.view(
            1, self.D, self.H, self.H
        )
        features = utils.standardise(features)
        abunds = self.convblock(features)
        abunds = self.sum_to_one(abunds)
        return abunds

    def forward(self, features):
        abunds = self.get_abundances(features)
        output = self.decoder(abunds)
        endmembers = self.get_endmembers()
        return endmembers, abunds, output

    def get_endmembers(self):
        endmembers = self.decoder.get_endmembers()
        return endmembers

class UnmixingFrom3FMs(nn.Module):
    def __init__(self, D, B, c, alpha_dofa, alpha_specvit, alpha_specaware):
        """
        Upsamples low res features from 3 FMs then estimates A_hat
        
        Args:
            D (int): The embed_dim
            B (int): The number of spectral bands in the hsi
            c (int): The number of endmembers to extract
            H (int): The size of the input hsi
            alpha (int): The size of the features
            n_features (int): The size of the list of features in the case of several extracted features

        """
        super(UnmixingFrom3FMs, self).__init__()
        self.D = D
        self.alpha_dofa = alpha_dofa
        self.alpha_specvit = alpha_specvit
        self.alpha_specaware = alpha_specaware
        self.B = B
        self.c = c
        self.H = 224
        self.H_specvit = 128

        # Upsampling features
        self.upsample_dofa = nn.Linear(self.alpha_dofa**2, self.H**2, bias=False)
        # self.upsample_specvit = nn.Linear(self.alpha_specvit**2, self.H_specvit**2, bias=False)
        self.upsample_specaware = nn.Linear(self.alpha_specaware**2, self.H**2, bias=False)
  
        # Upsampled features to abundances
        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

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

        """Feature regularisation"""

        if features_hr != None:
            l1 = nn.L1Loss()
            downsample = nn.AdaptiveAvgPool2d(features_lr[0].shape)
            features_down = downsample(features_hr)

            loss_features = W_feat * l1(utils.normalise(features_down), utils.normalise(features_lr))
        
        else:
            loss_features = 0

        loss = loss_sad + loss_ab + loss_mse + loss_features

        if channel_selector != None:
            loss += torch.linalg.norm(channel_selector, dim=0, ord=1)#/ len(channel_selector)

        if return_losses:
            return loss, loss_sad, loss_ab, loss_mse
        else:
            return loss

    def get_abundances(self, features_dofa, features_specaware):

        features_up_dofa = utils.oneD_to_2d(self.upsample_dofa(features_dofa)).unsqueeze(0)
        features_up_specaware = utils.oneD_to_2d(self.upsample_specaware(features_specaware)).unsqueeze(0)

        features_up = torch.cat((features_up_dofa, features_up_specaware), dim=1)
        # features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
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
 
class UnmixingFromFeaturesv2(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, n_features=1, upsampler="Linear", channel_selector=False, kernel_size=1):
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
        super(UnmixingFromFeaturesv2, self).__init__()
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
        elif upsampler == "Features_fusion":
            self.upsample = upsamplers.FeaturesFusionUpsampler(self.n_features*D, B, alpha, H, group_channels=True)
        else:
            raise "Unknown upsampler, must be one of [Linear, Features_fusion]"
        
        # Upsampled features to abundances
        self.abundance_estimator = Abundance_estimator(self.D, self.c, self.n_features, kernel_size=kernel_size)

        self.sum_to_one = Sum_to_one()

        self.num_queries = 50 * c
        self.query_embed = nn.Embedding(self.num_queries, B)
        self.weights = nn.Parameter(torch.ones((c, 50)))

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09, hypersigma=False):
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
        
        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

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
            # features = features.reshape(self.D, self.alpha*self.alpha)
            features_up = self.upsample(features)
        else:
            if "2" not in self.upsampler:
                features = utils.oneD_to_2d(features).unsqueeze(0)
            features_up = self.upsample(features, Y)
        features_up = features_up.view(
            1, self.n_features*self.D, self.H, self.H
        )
        # features_up = utils.oneD_to_2d(self.spectral_regul(features_up.flatten(2).permute(0, 2, 1)).permute(0, 2, 1))
        # features_up = utils.oneD_to_2d(self.spectral_regul(features_up.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1))

        features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
        A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.c, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = self.weights.unsqueeze(-1).repeat(1, 1, self.B) * query_embed_weight_split
        endmember_get = torch.mean(endmember_get, dim=1)
        return endmember_get.T
    
    def forward(self, features, Y):
        A_hat = self.get_abundances(features, Y)
        # Y_hat = self.decoder(A_hat)
        E_hat = self.get_endmembers()
        Y_hat = torch.einsum('bchw,lc->blhw', [A_hat, E_hat])

        return E_hat, A_hat, Y_hat
 
class UnmixingFromFeatures(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, n_features=1, channel_selector=False, kernel_size=1):
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

        # Upsampling features
        self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)
        
        # self.spectral_regul = nn.Linear(self.n_features*D, self.n_features*D)

        # Upsampled features to abundances
        self.abundance_estimator = Abundance_estimator(self.D, self.c, self.n_features, kernel_size=kernel_size)

        # self.smooth = nn.Sequential(
        #     nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.Softmax(dim=1),
        # )

        if channel_selector:
        # self.channel_selector = nn.Parameter(torch.ones(self.D))
            self.channel_selector = nn.Parameter(torch.ones(self.n_features))

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09, hypersigma=False, return_losses=False):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')
        mse2 = nn.MSELoss()
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        if hypersigma:
            loss_mse = W_mse * losses.hypersigma_mse(Y_gt, Y_hat)
        else:
            loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        # Endmember regul : distance to mean spectrum
        # B, c = E_hat.shape
        # mean_e = Y_gt.mean(dim=(2,3)).flatten(0).repeat(c).reshape(B, c)
        # loss_e = W_e * mse2(mean_e, E_hat)
        # loss_a = W_a * torch.norm(A_hat, p=0.5, dim=1).mean()

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        if return_losses:
            return loss, loss_sad, loss_ab, loss_tv_e, loss_mse
        else:
            return loss

    def get_abundances(self, features):

        if hasattr(self, "channem_selector"):
            if len(self.channel_selector) == self.n_features:
                weights = self.channel_selector.view(-1, 1, 1)
                features = features.view(self.n_features, self.D, self.alpha**2)
                features = features * weights
                features = features.view(-1, self.alpha**2)
            else:
                weights = self.channel_selector.view(-1, 1)
                features = features * weights

        features_up = self.upsample(features)
        features_up = features_up.view(
            1, self.n_features*self.D, self.H, self.H
        )

        features_up = utils.standardise(features_up)
        A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)

        return A_hat
    
    def get_endmembers(self):
        return self.decoder.get_endmembers()
    
    def forward(self, features):
        A_hat = self.get_abundances(features)
        Y_hat = self.decoder(A_hat)
        E_hat = self.decoder.get_endmembers()

        return E_hat, A_hat, Y_hat
 
class DeepTransEncoder(nn.Module):
    def __init__(self, B, c, H, embed_dim=512, patch_size=5):
        super(DeepTransEncoder, self).__init__()

        self.c = c
        self.H = H

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

        self.vtrans = transformer.ViT(image_size=H, patch_size=patch_size, embed_dim=(embed_dim*c), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(embed_dim, H ** 2),
        )
        
        self.smooth = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

    def forward(self, Y):
        abu_est = self.encoder(Y)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.c, -1)
        abu_est = self.upscale(cls_emb).view(1, self.c, self.H, self.H)
        abu_est = self.smooth(abu_est)

        return abu_est

class UnmixingFromFeaturesTrans(nn.Module):
    def __init__(self, D, B, c, H=224, alpha=None, patch_size=5, embed_dim=200):
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
        super(UnmixingFromFeaturesTrans, self).__init__()
        self.D = D
        self.alpha = alpha
        self.B = B
        self.c = c
        self.H = H

        # Upsampling features
        self.upsample = nn.Linear(self.alpha**2, self.H**2, bias=False)
        
        # Upsampled features to abundances
        self.abundance_estimator = DeepTransEncoder(D, c, H, patch_size=patch_size, embed_dim=embed_dim)

        self.sum_to_one = Sum_to_one()
        self.decoder = Decoder(B=B, c=c)

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    @staticmethod
    def loss(Y_gt, Y_hat, A_hat, E_hat, W_sad=1, W_ab=0.6, W_tv_e=3e-5, W_mse=0.09, hypersigma=False, return_losses=False):
        sad = losses.SADLoss()
        mse = nn.MSELoss(reduction='sum')
        mse2 = nn.MSELoss()
        
        loss_sad = W_sad * sad(Y_gt, Y_hat)
        loss_ab = W_ab * torch.sqrt(A_hat).mean()

        if hypersigma:
            loss_mse = W_mse * losses.hypersigma_mse(Y_gt, Y_hat)
        else:
            loss_mse = W_mse * mse(Y_gt, Y_hat)/(torch.norm(Y_gt)**2)

        """Abundances and endmembers regularisation"""

        # TV on endmembers (sum of difference between consecutive endmembers)
        loss_tv_e = W_tv_e * (torch.abs(E_hat[:, 1:] - E_hat[:, :-1]).sum())

        # Endmember regul : distance to mean spectrum
        # B, c = E_hat.shape
        # mean_e = Y_gt.mean(dim=(2,3)).flatten(0).repeat(c).reshape(B, c)
        # loss_e = W_e * mse2(mean_e, E_hat)
        # loss_a = W_a * torch.norm(A_hat, p=0.5, dim=1).mean()

        loss = loss_sad + loss_ab + loss_tv_e + loss_mse

        if return_losses:
            return loss, loss_sad, loss_ab, loss_tv_e, loss_mse
        else:
            return loss

    def get_abundances(self, features, Y):

        features_up = self.upsample(features)
        features_up = features_up.view(
            1, self.D, self.H, self.H
        )
        features_up = (features_up - features_up.mean())/ (1e-8 + features_up.std())
        A_hat = self.abundance_estimator(features_up)
        A_hat = self.sum_to_one(A_hat)

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
            raise "Unknown upsampler, must be one of [Linear, FiLM, FiLM_grouped, Features_fusion, Features_fusion_grouped]"

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(D, c, kernel_size=1, bias=False),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        # self.channel_selector = nn.Parameter(torch.ones(self.D))
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

class Weight_constraint(object):
    def __init__(self):
        pass
    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.clamp_(min=1e-7, max=1)

class Sum_to_one(nn.Module):
    def __init__(self, scale=1):
        super(Sum_to_one, self).__init__()
        self.scale = scale
    def forward(self, x):
        # print(x.max())
        x = F.softmax(self.scale * x, dim=1)
        return x

class Abundance_estimator(nn.Module):
    def __init__(self, D, c, n_features, kernel_size=1):
        super(Abundance_estimator, self).__init__()

        self.abundance_estimator = nn.Sequential(
            nn.Conv2d(n_features*D, c, kernel_size=kernel_size, bias=False, padding="same"),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(c),
            nn.Dropout(0.2)
        )

        # self.spectral_regul = nn.Linear(n_features*D, n_features*D)
    
    def forward(self, up_feat):

        # up_feat = utils.oneD_to_2d(self.spectral_regul(up_feat.flatten(2).permute(0,2,1)).permute(0,2,1))
        A_hat = self.abundance_estimator(up_feat)

        return A_hat

class Decoder(nn.Module):
    def __init__(self, c, B, kernel_size=1):
        super(Decoder, self).__init__()
        self.B = B
        self.c = c

        padding = kernel_size //2
        self.decoder = nn.Conv2d(in_channels=c, out_channels=B,
                                kernel_size=kernel_size,stride=1,
                                padding=padding, bias=False)

    def forward(self, code):

        code = self.decoder(code)
        
        return code

    def get_endmembers(self):

        return self.decoder.weight.data.squeeze([2, 3])