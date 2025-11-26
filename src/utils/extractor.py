import torch
import torch.linalg as LA
from sklearn.cluster import KMeans
# from cvxopt import matrix, solvers
import numpy as np
import time

def estimate_snr(Y, r_m, x):
    N, B = Y.shape # B number of bands (channels), N number of pixels
    c, N = x.shape  # c number of endmembers (reduced dimension)
    P_y = torch.sum(Y**2) / float(N)
    P_x = torch.sum(x**2) / float(N) + torch.sum(r_m**2)
    snr_est = 10 * torch.log10((P_x - c / B * P_y) / (P_y - P_x))

    return snr_est

def VCA(Y, c, seed=None, snr_input=0):
    """
    Vertex Component Analysis algorithm by Jose M. P. Nascimento and Jose M. B. Dias
    
    Args:
        Y: input HSI to extract endmembers from (shape (B, h, w) or (B, N))
        c (int): the number of endmembders to extract
        snr_input: the snr of the input image (default: 0)
    """
    if Y.dim()!= 2:
        B, h, w = Y.shape
        N = h*w
        Y = Y.reshape(B, N)
    else:
        B, N = Y.shape
    
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
    
    if snr_input == 0:
        y_m = torch.mean(Y, dim=1, keepdim=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = LA.svd(torch.matmul(Y_o, Y_o.T) / float(N))[0][:, :c]  # computes the R-projection matrix
        x_c = torch.matmul(Ud.T, Y_o)  # project the zero-mean data onto c-subspace

        SNR = estimate_snr(Y, y_m, x_c)

        print(f"input SNR estimated = {SNR}[dB]")
    else:
        SNR = snr_input
        print(f"input SNR = {SNR}[dB]\n")

    SNR_th = 15 + 10 * torch.log10(torch.tensor(c))

    if SNR < SNR_th:

        d = c - 1
        if snr_input == 0:  # it means that the projection is already computed
            Ud = Ud[:, :d]
        else:
            y_m = torch.mean(Y, dim=1, keepdim=True)
            Y_o = Y - y_m  # data with zero-mean

            Ud = LA.svd(torch.matmul(Y_o, Y_o.T) / float(N))[0][:, :d]  # computes the c-projection matrix
            x_c = torch.matmul(Ud.T, Y_o)  # project thezeros mean data onto c-subspace

        Yc = torch.matmul(Ud, x_c[:d, :]) + y_m  # again in dimension c

        x = x_c[:d, :]  #  x_c =  Ud.T * Y_o is on a R-dim subspace
        b = torch.max(torch.sum(x**2, dim=0)) ** 0.5
        y = torch.vstack((x, b * torch.ones((1, N))))
    else:

        d = c
        Ud = LA.svd(torch.matmul(Y, Y.T) / float(N))[0][:, :c]  # computes the c-projection matrix

        x_c = torch.matmul(Ud.T, Y)
        Yc = torch.matmul(Ud, x_c[:d, :])  # again in dimension b (note that x_c has no null mean)

        x = torch.matmul(Ud.T, Y)
        u = torch.mean(x, dim=1, keepdim=True)  # equivalent to  u = Ud.T * r_m
        y = x / torch.matmul(u.T, x)

    #############################################
    # VCA algorithm
    #############################################

    indices = torch.zeros((c), dtype=torch.long)
    A = torch.zeros((c, c))
    A[-1, 0] = 1
    
    for i in range(c):
        w = torch.rand(size=(c, 1))
        f = w - torch.matmul(A, LA.pinv(A) @ w)
        f = f / LA.norm(f)

        v = torch.matmul(f.T, y)

        indices[i] = torch.argmax(torch.abs(v))
        A[:, i] = y[:, indices[i]]  # same as x(:,indice(i))
        
    E = Yc[:, indices] 

    print(f"Indices chosen to be the most pure: {indices}")

    return E
    
def FCLS(Y, E):
    """
    Performs fully constrained least squares to obtain the abundance matrices from Y and E

    Args:
        Y: HSI data matrix (B x N)
        E: Matrix of endmembers (B x c)
    """
    if Y.dim() != 2:
        Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2])
    B1, N = Y.shape
    B2, c = E.shape

    if B1 != B2:
        raise ValueError("M and U must have the same number of spectral bands.")

    eet = E.T @ E
    eet_inv = torch.linalg.inv(eet)
    eet_inv_eT = eet_inv @ E.T  # (c x B)

    # Unconstrained least squares for all pixels: (c x N)
    abd_ls = eet_inv_eT @ Y

    # Apply sum-to-one constraint
    ones_row = torch.ones(1, c, device=Y.device)
    ones_col = torch.ones(c, 1, device=Y.device)
    scaling = ones_row @ eet_inv @ ones_col
    sum_ls = ones_row @ abd_ls  # (1 x N)
    abd_fcls = abd_ls - eet_inv @ ones_col @ (1 / scaling) * (sum_ls - 1)

    # Clip negative abundances to zero
    abd_clipped = torch.clamp(abd_fcls, min=0)
    sum_abd = torch.sum(abd_clipped, dim=0, keepdim=True)  # (1 x N)
    
    # Avoid division by zero (if all abundances are zero for a pixel)
    sum_abd[sum_abd == 0] = 1  # Set to 1 to avoid NaN
    
    A = abd_clipped / sum_abd

    return A

def unmix(y, c):
    """
    Unmixes the input HSI y into c endmembers by applying VCA + FCLS
    
    Args:
        y (torch.tensor): the input hsi tensor (shape (B, h, w) or (B, N))
        c (int): the number of endmembers to unmix
    """
    
    E = VCA(y, c)
    A = FCLS(y, E)
    
    return E, A
    