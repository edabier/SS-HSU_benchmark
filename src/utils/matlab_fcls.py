import torch
from math import *

def simpleY_col_proj(Y):
    Y = Y.T        # now: N Y D
    N, D = Y.shape
    
    # Sort descending row-wise
    Y = torch.sort(Y, dim=1, descending=True).values
    csum = torch.cumsum(Y, dim=1) - 1.0
    denom = torch.arange(1, D+1, device=Y.device, dtype=Y.dtype)
    Ytmp = csum / denom
    mask = Y > Ytmp
    k = mask.sum(dim=1) - 1   # zero-indeYed
    theta = Ytmp[torch.arange(N), k]

    Yproj = torch.clamp(Y - theta.unsqueeze(1), min=0)

    return Yproj.t()

def FCLS_2(Y, E, delta=1e-6, inner_iter=500, alpha0=0.05):
    """
    Solves min_{A >= 0} ||Y - EA||_F^2
    """

    E = E.to_dense() if E.is_sparse else E
    b, n = Y.shape
    _, c = E.shape

    EtE = E.t() @ E
    EtY = E.t() @ Y

    # Initialization
    A = torch.clamp(EtY, min=0)

    # Lipschitz constant L = ||EtE||_2
    L = torch.linalg.norm(EtE, 2)

    alpha = [alpha0]

    A = simpleY_col_proj(A)   # simpleYColProj operates on columns -> rows of H

    Y = A.clone()
    i = 1
    eps0 = 0
    eps = 1e9

    while i <= inner_iter and eps >= delta * eps0:
        Ap = A.clone()

        # FGM coefficients
        a_i = alpha[i-1]
        a_neYt = (sqrt(a_i**4 + 4*a_i**2) - a_i**2) / 2
        alpha.append(a_neYt)
        beta = a_i * (1 - a_i) / (a_i**2 + a_neYt)

        # Gradient step + projection
        v = Y - (EtE @ Y - EtY) / L

        A = simpleY_col_proj(A)

        # Acceleration step
        Y = A + beta * (A - Ap)

        # stopping rule
        diff = torch.norm(A - Ap, p='fro')
        if i == 1:
            eps0 = diff
        eps = diff

        i += 1

    return A, EtE, EtY