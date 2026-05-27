import torch.nn as nn
import torch
import torch.nn.functional as F

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        return self.conv_transpose(x)

class x2UpsampleBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels=c,
            out_channels=c,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(c)  # or InstanceNorm2d(C)
        self.activate = nn.LeakyReLU(0.01)

        # Initialize weights
        nn.init.kaiming_normal_(
            self.upsample.weight,
            mode='fan_out',
            nonlinearity='leaky_relu'
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.norm(x)
        x = self.activate(x)
        return x

class FusedFeaturesUpsampler(nn.Module):
    def __init__(self, C, B, alpha, H):
        super().__init__()
        self.C = C
        self.B = B
        self.alpha = alpha
        self.H = H

        # Step 1: Upsample low-res tensor (C channels) to (batch, C, H, H)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=H // alpha, mode='bilinear', align_corners=False),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
        )

        # Step 2: Extract features from high-res tensor (B channels)
        self.extract_hr = nn.Conv2d(B, C, kernel_size=1)  # Project B to C channels

        # Step 3: Fuse upsampled low-res and high-res features
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=3, padding=1),  # Concatenate and fuse
            nn.BatchNorm2d(C),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x_lr, x_hr):
        # x_lr: (batch, C, alpha, alpha)
        # x_hr: (batch, B, H, H)

        # Upsample low-res tensor
        x_upsampled = self.upsample(x_lr)  # (batch, C, H, H)

        # Extract features from high-res tensor
        x_hr_feat = self.extract_hr(x_hr)  # (batch, C, H, H)

        # Fuse by concatenation
        x_fused = torch.cat([x_upsampled, x_hr_feat], dim=1)  # (batch, 2*C, H, H)
        x_fused = self.fuse(x_fused)  # (batch, C, H, H)

        return x_fused

class CrossChannelAttention(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.query = nn.Conv2d(C, C, kernel_size=1)
        self.key = nn.Conv2d(C, C, kernel_size=1)
        self.value = nn.Conv2d(C, C, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling

    def forward(self, x_lr, x_hr):
        # x_lr: (batch, C, H, H) (upsampled low-res)
        # x_hr: (batch, C, H, H) (high-res features)
        batch, C, H, W = x_lr.shape

        # Project to query, key, value
        q = self.query(x_lr).view(batch, C, -1).permute(0, 2, 1)  # (batch, H*W, C)
        k = self.key(x_hr).view(batch, C, -1)  # (batch, C, H*W)
        v = self.value(x_hr).view(batch, C, -1)  # (batch, C, H*W)

        # Compute attention
        attention = F.softmax(torch.bmm(q, k), dim=-1)  # (batch, H*W, H*W)
        out = torch.bmm(v, attention.permute(0, 2, 1))  # (batch, C, H*W)
        out = out.view(batch, C, H, W)

        # Scale and add residual
        return x_lr + self.gamma * out

class AttentionUpsampler(nn.Module):
    def __init__(self, C, B, alpha, H):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=H // alpha, mode='bilinear', align_corners=False),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
        )
        self.extract_hr = nn.Conv2d(B, C, kernel_size=1)
        self.attention = CrossChannelAttention(C)

    def forward(self, x_lr, x_hr):
        x_upsampled = self.upsample(x_lr)  # (batch, C, H, H)
        x_hr_feat = self.extract_hr(x_hr)  # (batch, C, H, H)
        x_fused = self.attention(x_upsampled, x_hr_feat)
        return x_fused

class FiLMLayer(nn.Module):
    def __init__(self, c, B):
        super().__init__()
        self.gamma = nn.Conv2d(B, c, kernel_size=1)  # Scale
        self.beta = nn.Conv2d(B, c, kernel_size=1)   # Shift

        nn.init.constant_(self.gamma.weight, 1.0)
        nn.init.constant_(self.beta.weight, 0.0)

    def forward(self, x, x_hr):
        # x: (batch, c, H, W)
        # x_hr: (batch, B, H, W)
        gamma = self.gamma(x_hr)  # (batch, c, H, W)
        beta = self.beta(x_hr)    # (batch, c, H, W)
        return x * gamma + beta

class FiLMUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, B, alpha, H, group_channels=False):
        super().__init__()
        
        if group_channels:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(H // alpha), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            )
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=(H // alpha), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            )
        self.film = FiLMLayer(out_channels, B)

    def forward(self, x_lr, x_hr):
        x_upsampled = self.upsample(x_lr)  # (batch, c, H, H)
        x_fused = self.film(x_upsampled, x_hr)
        return x_fused
    