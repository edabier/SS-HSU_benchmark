import torch
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur
from math import *
import random

"""
A set of data augmentation methods to generate positive pairs for contrastive learning
"""

def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return torch.cond(
      torch.less(
          torch.random.uniform([], minval=0, maxval=1, dtype=torch.float32),
          torch.cast(p, torch.float32)), lambda: func(x), lambda: x)

def crop_and_resize(y, r, return_position=False):
    """
    Crops and resize an input HSI
    
    /!\ supposes that width = y.shape[1] and height = y.shape[2]
    
    Args:
        y (torch.tensor): the input tensor to be cropped
        r (int): the scale ratio
        return_position (bool, optional): if True, returns the coordinates and size of the part of crop in the original image
    """
    
    c       = y.shape[0]
    width   = y.shape[1]
    height  = y.shape[2]
    
    x0 = random.randint(1, width - floor(r*width))
    y0 = random.randint(1, height - floor(r*height))
    
    y_crop  = y[:, x0:x0+floor(r*width), y0:y0+floor(r*height)]
    new_y   = F.resize(y_crop, [width, height])
    
    if return_position:
        return new_y, (x0, y0), (floor(r*width), floor(r*height))
    else:
        return new_y

def flip(y, horizontal_p=0.5, both_p=0.1):
    """
    Randomly flips horizontaly or verticaly an input HSI
    Can also flip in both directions with 10% chance
    
    Args:
        y (torch.tensor): the input tensor to be flipped
        horizontal_p (float, optional): the probability with which to flip horizontaly or vericaly (default: 0.5)
        both_p (float, optional): the probability to flip in both directions (default: 0.1)
    """
    horizontal  = random.random() < horizontal_p
    both        = random.random() < both_p
    
    if both:
        new_y = torch.flip(y, [1,2])
    else:
        if horizontal:
            new_y = torch.flip(y, [2])  
        else:
            new_y = torch.flip(y, [1])      
    
    return new_y    

def blur(y, r, sigma):
    """
    Applies gaussian blurring to an input HSI
    
    Args:
        y (torch.tensor): the input tensor to be blurred
        r (int): the gaussian convolutional kernel size
        sigma (float): the gaussian variance
    """
    
    gb = GaussianBlur(r, sigma)
    return gb(y)

def spectral_variability(y, c):
    """
    Applies a randomly drawn piece-wise affine function to the tensor's bands
    
    Args:
        y (torch.tensor): the input tensor to be transformed
        c (float): the variability coefficient
    """
    
    # Randomly draw the piece-wise affine function points
    y0 = torch.rand(y.shape[1], y.shape[2], device=y.device) * c + (1 - c/2)
    y1 = torch.rand(y.shape[1], y.shape[2], device=y.device) * c + (1 - c/2)
    y2 = torch.rand(y.shape[1], y.shape[2], device=y.device) * c + (1 - c/2)
    x = torch.floor(y.shape[0]/2 + torch.floor(y.shape[0] * torch.randn(y.shape[1], y.shape[2], device=y.device) / 3)).clamp(0, y.shape[0])
    
    # Create a tensor of band indices (t) for each pixel
    t = torch.arange(y.shape[0], device=y.device).view(y.shape[0], 1, 1).float()
    
    return y * torch.where(
        t <= x,
        y0 + (y1 - y0) * t / x,
        y1 + (y2 - y1) * (t - x) / (y.shape[0] - x)
    )
    
def spectral_jitter(y):
    """
    Applies spectral jittering to an input HSI
    
    Args:
        y (torch.tensor): the input tensor to be transformed
    """
    
    pass