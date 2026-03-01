import torch

def quadratic_penalty(error, tol=0.0):
    """
    error: tensor of arbitrary shape
    tol: tolerance; 0 -> pure L2
    returns: penalty tensor of same shape
    """
    # |error|
    abs_err = error.abs()
    # ReLU(|e| - tol)^2
    return torch.relu(abs_err - tol) ** 2
