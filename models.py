import torch.nn as nn
import torch

"""
Autoencoders
"""

class CNNAEU(nn.Module):
    """
    Defines a CNN autoencoder based on CNNAEU
    
    Args:
        n (int): the input patch size
        f (int): the kernel size of the decoder conv layer
    """
    def __init__(self, n, in_channels, endmembers, f):
        super(CNNAEU, self).__init__()
        
        num_channels = None
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=endmembers, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=endmembers, out_channels=in_channels, kernel_size=f)
        
        # We use the ReLU activation for the ANC and the LReLU for better performance
        # We use the same architecture as CNNAEU
        self.lrelu  = nn.LeakyReLU(negative_slope=0.1)
        self.linear = nn.Linear(in_features=n*n*in_channels, out_features=n*n*in_channels)
        self.bn     = nn.BatchNorm2d(num_features=num_channels)
        self.dropout = nn.Dropout2d(p=0.2)
        
        # We use a softmax activation for the ASC
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        # First encoder conv
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        # second encoder conv
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        # Decoder conv
        x = self.conv2(x)
        out = self.linear(x) 
        
        return out
    
class CNNAE_linear(nn.Module):
    """
    Defines a CNN autoencoder with linear decoder serving as the endmember estimation
    
    Args:
        n (int): the input patch size
    """
    def __init__(self, n, in_channels, endmembers, f):
        super(CNNAE_linear, self).__init__()
        
        num_channels = None
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=endmembers, kernel_size=1)
        self.fc3   = nn.Linear(in_features=n*n*48, out_features=n*n*in_channels)
        
        # We use the ReLU activation for the ANC and the LReLU for better performance
        # We use the same architecture as CNNAEU
        self.lrelu  = nn.LeakyReLU(negative_slope=0.1)
        self.relu   = nn .ReLU()
        self.bn     = nn.BatchNorm2d(num_features=num_channels)
        self.dropout = nn.Dropout2d(p=0.2)
        
        # We use a softmax activation for the ASC
        self.softmax = nn.Softmax()

    def forward(self, x):
        
        # First encoder conv
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        # second encoder conv
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.bn(x)
        x = self.dropout(x)
        
        # Decoder conv
        x = self.fc3(x)
        out = self.relu(x)
        
        return out

class TransformerAE(nn.Module):
    pass
    
    
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
