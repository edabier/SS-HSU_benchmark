import torch
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur
from sklearn.cluster import KMeans
import numpy as np
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
    
    Supposes that height = y.shape[1] and width = y.shape[2]
    
    Args:
        y (torch.tensor): the input tensor to be cropped
        r (int): the scale ratio
        return_position (bool, optional): if True, returns the coordinates and size of the part of crop in the original image
    """
    
    c, h, w = y.shape
    
    x0 = random.randint(1, h - floor(r*h))
    y0 = random.randint(1, w - floor(r*w))
    
    y_crop  = y[:, x0:x0+floor(r*h), y0:y0+floor(r*w)]
    new_y   = F.resize(y_crop, [h, w])
    
    if return_position:
        return new_y, (x0, y0), (floor(r*h), floor(r*w))
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
        new_y = torch.flip(y, [0,1])
    else:
        if horizontal:
            new_y = torch.flip(y, [1])  
        else:
            new_y = torch.flip(y, [0])      
    
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

def spectral_variability(y, c_var):
    """
    Applies a randomly drawn piece-wise affine function to the tensor's bands
    
    Args:
        y (torch.tensor): the input tensor to be transformed
        c_var (float): the variability coefficient
    """
    h, w, c = y.shape
    
    # Randomly draw the piece-wise affine function points
    y0 = torch.rand(h, w, device=y.device) * c_var + (1 - c_var/2)
    y1 = torch.rand(h, w, device=y.device) * c_var + (1 - c_var/2)
    y2 = torch.rand(h, w, device=y.device) * c_var + (1 - c_var/2)
    x = torch.floor(c/2 + torch.floor(c * torch.randn(h, w, device=y.device) / 3)).clamp(0, c)
    
    # Create a tensor of band indices (t) for each pixel
    t = torch.arange(c, device=y.device).view(c, 1, 1).float()
    
    return y * torch.where(
        t <= x,
        y0 + (y1 - y0) * t / x,
        y1 + (y2 - y1) * (t - x) / (c - x)
    )
    
def spectral_jitter(y):
    """
    Applies spectral jittering to an input HSI
    
    Args:
        y (torch.tensor): the input tensor to be transformed
    """
    pass

"""
Utils functions for the training Dataset generation
"""

def remove_duplicates(lib, tol=1e-3):
    """
    Removes duplicates from spectra library
    """
    unique_spectra = []
    
    for spec in lib.T:
        if not any(torch.linalg.norm(spec - uniq) < tol for uniq in unique_spectra):
            unique_spectra.append(spec)
    return torch.stack(unique_spectra).T

def group_spectra_kmeans(spectra, n_clusters, seed=42):
    """
    Groups spectra using scikit-learn's Kmeans algorithm.
    """
    spectra = spectra.numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(spectra)
    centers = kmeans.cluster_centers_
    
    return centers, labels

def group_spectra_by_cluster(lib, memberships):
    """
    Groups spectra by cluster by attributing the cluster by maximum belonging degree.
    
    Args:
        lib: spectra library of shape (n_spectra, c)
        memberships: belonging matrix of shape (n_clusters, n_spectra)
    
    Returns:
        groups: groups[i] contains cluster's i spectras of shape (nb_spectra_in_cluster, c).
    """
    # Attribution to cluster using argmax on the cluster's dimension
    n_clusters = memberships.max().item() + 1
    groups = []
    for i in range(n_clusters):
        indices = np.where(memberships == i)[0]
        group = torch.stack([lib[:, idx] for idx in indices], dim=1)
        groups.append(group)
    return groups

def augment_spectrum(spectrum, c_var=0.4):
    """
    Applies a randomly drawn piece-wise affine function to the tensor's bands
    
    Args:
        spectrum (torch.tensor): the input spectrum to be transformed
        c_var (float): the variability coefficient
    """
    c = len(spectrum)
    # Randomly draw the piece-wise affine function points
    y0 = torch.rand(1) * c_var + (1 - c_var/2)
    y1 = torch.rand(1) * c_var + (1 - c_var/2)
    y2 = torch.rand(1) * c_var + (1 - c_var/2)
    x = torch.floor(c/2 + torch.floor(c * torch.randn(1) / 3)).clamp(0, c)
    
    # Create a tensor of band indices (t) for each pixel
    t = torch.arange(c).float()
    
    return spectrum * torch.where(
        t <= x,
        y0 + (y1 - y0) * t / x,
        y1 + (y2 - y1) * (t - x) / (c - x)
    )
