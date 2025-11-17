import time
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import transformer
from sklearn.feature_extraction.image import extract_patches_2d
import tqdm

"""
Base
"""
class UnmixingModel:
    def __init__(self):
        self.time = 0

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    def print_time(self):
        return f"{self} took {self.time:.2f}s"
    
class BlindUnmixingModel(UnmixingModel):
    def __init__(
        self,
    ):
        super().__init__()

    def compute_endmembers_and_abundances(
        self,
        Y,
        p,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(f"Solver is not implemented for {self}")


"""
Autoencoders
"""

class CNNAE_linear(nn.Module, BlindUnmixingModel):
    """
    Adaptation of the CNNAEU implementation from the HySUPP repo
    """
    def __init__(
        self,
        scale=3.0,
        epochs=320,
        lr=0.0003,
        batch_size=15,
        patch_size=40
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }

        self.scale = scale
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.L,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.p, kernel_size=1, bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(self.p),
            nn.Dropout2d(p=0.2),
        )

        self.decoder = nn.Linear(in_features=self.H*self.W*self.p, out_features=self.H*self.W*self.L)

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        return abund, x_hat

    @staticmethod
    def loss(target, output):
        assert target.shape == output.shape

        dot_product = (target * output).sum(dim=1)
        target_norm = target.norm(dim=1)
        output_norm = output.norm(dim=1)
        sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
        return sad_score.mean()

    def unmix(self, Y, p, H, W, seed=0):
        tic = time.time()

        L, N = Y.shape
        # Hyperparameters
        self.L = L  # number of channels
        self.p = p  # number of dictionary atoms
        self.H = H  # number of lines
        self.W = W  # number of samples per line

        self.num_patches = int(250 * self.H * self.W * self.L / (307 * 307 * 162))

        self.init_architecture(seed=seed)

        num_channels, h, w = self.L, self.H, self.W

        Y_numpy = Y.reshape((num_channels, h, w)).transpose((1, 2, 0))

        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=self.num_patches,
            patch_size=(self.patch_size, self.patch_size),
        )
        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        # Send model to GPU
        self = self.to(self.device)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            input_patches,
            batch_size=self.batch_size,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        for ee in progress:

            running_loss = 0
            for ii, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()

                _, outputs = self(batch)

                # Reshape data
                loss = self.loss(batch, outputs)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            progress.set_postfix_str(f"loss={running_loss:.3e}")

        # Get final abundances and endmembers
        self.eval()

        Y_eval = torch.Tensor(Y.reshape((1, num_channels, h, w))).to(self.device)

        abund, _ = self(Y_eval)

        Ahat = abund.detach().cpu().numpy().reshape(self.p, self.H * self.W)
        Ehat = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

        self.time = time.time() - tic

        return Ehat, Ahat

class CNNAEU(nn.Module, BlindUnmixingModel):
    """
    CNNAEU implementation from the HySUPP repo
    """
    def __init__(
        self,
        scale=3.0,
        epochs=320,
        lr=0.0003,
        batch_size=15,
        patch_size=40
    ):
        super().__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu",
        )

        self.lrelu_params = {
            "negative_slope": 0.02,
            "inplace": True,
        }

        self.scale = scale
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patch_size = patch_size

    def init_architecture(
        self,
        seed,
    ):
        # Set random seed
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(
                self.L,
                48,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=False,
            ),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(48),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(48, self.p, kernel_size=1, bias=False),
            nn.LeakyReLU(**self.lrelu_params),
            nn.BatchNorm2d(self.p),
            nn.Dropout2d(p=0.2),
        )

        self.decoder = nn.Conv2d(
            self.p,
            self.L,
            kernel_size=11,
            padding=5,
            padding_mode="reflect",
            bias=False,
        )

    def forward(self, x):
        code = self.encoder(x)
        abund = F.softmax(code * self.scale, dim=1)
        x_hat = self.decoder(abund)
        return abund, x_hat

    @staticmethod
    def loss(target, output):
        assert target.shape == output.shape

        dot_product = (target * output).sum(dim=1)
        target_norm = target.norm(dim=1)
        output_norm = output.norm(dim=1)
        sad_score = torch.clamp(dot_product / (target_norm * output_norm), -1, 1).acos()
        return sad_score.mean()

    def unmix(self, Y, p, H, W, seed=0):
        tic = time.time()

        L, N = Y.shape
        # Hyperparameters
        self.L = L  # number of channels
        self.p = p  # number of dictionary atoms
        self.H = H  # number of lines
        self.W = W  # number of samples per line

        self.num_patches = int(250 * self.H * self.W * self.L / (307 * 307 * 162))

        self.init_architecture(seed=seed)

        num_channels, h, w = self.L, self.H, self.W

        Y_numpy = Y.reshape((num_channels, h, w)).transpose((1, 2, 0))

        input_patches = extract_patches_2d(
            Y_numpy,
            max_patches=self.num_patches,
            patch_size=(self.patch_size, self.patch_size),
        )
        input_patches = torch.Tensor(input_patches.transpose((0, 3, 1, 2)))

        # Send model to GPU
        self = self.to(self.device)
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        # Dataloader
        dataloader = torch.utils.data.DataLoader(
            input_patches,
            batch_size=self.batch_size,
            shuffle=True,
        )

        progress = tqdm(range(self.epochs))
        for ee in progress:

            running_loss = 0
            for ii, batch in enumerate(dataloader):
                batch = batch.to(self.device)
                optimizer.zero_grad()

                _, outputs = self(batch)

                # Reshape data
                loss = self.loss(batch, outputs)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            progress.set_postfix_str(f"loss={running_loss:.3e}")

        # Get final abundances and endmembers
        self.eval()

        Y_eval = torch.Tensor(Y.reshape((1, num_channels, h, w))).to(self.device)

        abund, _ = self(Y_eval)

        Ahat = abund.detach().cpu().numpy().reshape(self.p, self.H * self.W)
        Ehat = self.decoder.weight.data.mean((2, 3)).detach().cpu().numpy()

        self.time = time.time() - tic

        return Ehat, Ahat

class Transformer_AE(nn.Module):
    def __init__(self, P, L, size, patch, dim):
        super(Transformer_AE, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        self.encoder = nn.Sequential(
            nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
                                      heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            nn.Linear(dim, size ** 2),
        )
        
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )

    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        abu_est = self.encoder(x)
        cls_emb = self.vtrans(abu_est)
        cls_emb = cls_emb.view(1, self.P, -1)
        abu_est = self.upscale(cls_emb).view(1, self.P, self.size, self.size)
        abu_est = self.smooth(abu_est)
        re_result = self.decoder(abu_est)
        return abu_est, re_result


"""
Unrolling
"""

class NALMU_block(nn.Module):
    pass

class NALMU(nn.Module):
    pass

class RALMU_block(nn.Module):
    pass

class RALMU(nn.Module):
    pass
