import torch.nn as nn
import torch
import torch.nn.functional as F

class FeaturesFusionUpsampler(nn.Module):
    def __init__(self, C, B, alpha, H, group_channels=False):
        super().__init__()
        self.C = C
        self.B = B
        self.alpha = alpha
        self.H = H

        if group_channels:

            # Step 1: Upsample low-res tensor (C channels) to (batch, C, H, H)
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=H // alpha, mode='bilinear', align_corners=False),
                nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C),
            )

            # Step 2: Extract features from high-res tensor (B channels)
            self.extract_hr = nn.Conv2d(B, C, kernel_size=1)  # Project B to C channels

            # Step 3: Fuse upsampled low-res and high-res features
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * C, C, kernel_size=3, padding=1, groups=C),  # Concatenate and fuse
                nn.BatchNorm2d(C),
                nn.LeakyReLU(0.01),
            )
        
        else:

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

class FeaturesFusionUpsampler2(nn.Module):
    """
    Input features are already upsampled by shifts, no bilinear upsampling
    """
    def __init__(self, C, B, alpha, H, group_channels=False):
        super().__init__()
        self.C = C
        self.B = B
        self.alpha = alpha
        self.H = H

        if group_channels:

            # Step 1: Upsample low-res tensor (C channels) to (batch, C, H, H)
            self.upsample = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)

            # Step 2: Extract features from high-res tensor (B channels)
            self.extract_hr = nn.Conv2d(B, C, kernel_size=1)  # Project B to C channels

            # Step 3: Fuse upsampled low-res and high-res features
            self.fuse = nn.Sequential(
                nn.Conv2d(2 * C, C, kernel_size=3, padding=1, groups=C),  # Concatenate and fuse
                nn.BatchNorm2d(C),
                nn.LeakyReLU(0.01),
            )
        
        else:

            # Step 1: Upsample low-res tensor (C channels) to (batch, C, H, H)
            self.upsample = nn.Conv2d(C, C, kernel_size=3, padding=1)

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
