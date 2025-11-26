import torch
import numpy as np
import os

def save_checkpoint(data, filename):
    torch.save(data, filename)

def load_checkpoint(filename, model, optimizer):
    """
    Checks if a checkpoint exists, and if so, load the model to this checkpoint
    and returns the epoch at which to restart the training
    
    Args:
        filename (str): where to look for the checkpoint
    """
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        nbEpochsPretrained = checkpoint['epoch']
        model_state = checkpoint['model_state']
        optimizer_state = checkpoint['optimizer_state']
    
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
        return nbEpochsPretrained
    else:
        return 0
