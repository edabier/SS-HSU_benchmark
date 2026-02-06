import torch
import torch.linalg as LA
from math import *
from sklearn.cluster import KMeans
# from cvxopt import matrix, solvers
import numpy as np
import time

import src.utils.utils as utils

def normalize_endmembers(e):
    """
    Normalizes the values of the endmembers' spectra to lie in [0,1]
    
    Args:
        e: the endmember matrix to normalize
    """
    batched = e.dim() == 3
    
    if batched:
        return e/ torch.max(e, dim=1).values
    else:
        return e/ torch.max(e, dim=0).values

def estimate_snr(Y, r_m, x):
    B, N = Y.shape # B number of bands (channels), N number of pixels
    c, N = x.shape  # c number of endmembers (reduced dimension)
    P_y = torch.sum(Y**2) / float(N)
    P_x = torch.sum(x**2) / float(N) + torch.sum(r_m**2)
    snr_est = 10 * torch.log10((P_x - c / B * P_y) / (P_y - P_x))

    return snr_est

def batched_estimate_snr(Y, r_m, x):
    batch, B, N = Y.shape # B number of bands (channels), N number of pixels
    batch, c, N = x.shape  # c number of endmembers (reduced dimension)
    P_y = torch.sum(Y**2) / float(N)
    P_x = torch.sum(x**2) / float(N) + torch.sum(r_m**2)
    snr_est = 10 * torch.log10((P_x - c / B * P_y) / (P_y - P_x))

    return snr_est

def Eucli_dist(x,y):
    a = torch.subtract(x, y)
    return a.T @ a

class SiVM():
    def __init__(self):
        super().__init__()

    @staticmethod
    def Eucli_dist(x, y):
        a = np.subtract(x, y)
        return np.dot(a.T, a)

    def extract_endmembers(self, Y, p):

        x, p = Y, p

        [D, N] = x.shape
        # If no distf given, use Euclidean distance function
        Z1 = np.zeros((1, 1))
        O1 = np.ones((1, 1))
        # Find farthest point
        d = np.zeros((p, N))
        index = np.zeros((p, 1))
        V = np.zeros((1, N))
        ZD = np.zeros((D, 1))
        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), ZD)

        index = np.argmax(d[0, :])

        for i in range(N):
            d[0, i] = self.Eucli_dist(x[:, i].reshape(D, 1), x[:, index].reshape(D, 1))

        for v in range(1, p):
            D1 = np.concatenate(
                (d[0:v, index].reshape((v, index.size)), np.ones((v, 1))), axis=1
            )
            D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
            D4 = np.concatenate((D1, D2), axis=0)
            D4 = np.linalg.inv(D4)

            for i in range(N):
                D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
                V[0, i] = np.dot(np.dot(D3.T, D4), D3)

            index = np.append(index, np.argmax(V))
            for i in range(N):
                d[v, i] = self.Eucli_dist(
                    x[:, i].reshape(D, 1), x[:, index[v]].reshape(D, 1)
                )

        per = np.argsort(index)
        index = np.sort(index)
        d = d[per, :]
        E = x[:, index]
        return E

def SiVM(Y, c, E_gt=None): 
    """ 
    SiVM endmember extractor based on UnDIP's repository 

    Args: 
        Y: input HSI to extract endmembers from (shape (B, N) or (B, H, W), no batch) 
        c (int): the number of endmembders to extract 
        E_gt (optional): if set, used to reorder the extracted endmembers to match E_gt 
    """ 
    dev = Y.device

    if Y.dim()!= 2:
        if Y.dim() ==4:
            Y = Y[0]
        B, h, w = Y.shape
        N = h*w
        Y = Y.reshape(B, N)
    else:
        B, N = Y.shape
    
    Vh, S, U = torch.linalg.svd(Y, full_matrices=False)
    PC = torch.diag(S) @ U 
    Yp = Vh[:, :c] @ PC[:c, :] 
    d = torch.zeros((c, N), device=dev) # distance matrix 
    I = [] # endmembers indices 
    
    # First endmember: farthest from origin 
    d[0] = torch.sum(Y**2, dim=0) 
    I.append(torch.argmax(d[0, :])) 
    
    for v in range(1, c): 
        E = Yp[:, I] # Selected endmembers (shape: B x v) 
        P = E @ torch.linalg.pinv(E.T @ E) @ E.T 
        residual = Yp - P @ Yp 
        d[v] = torch.sum(residual**2, dim=0) # Squared orthogonal distance 
        d[v, I] = -torch.inf 
        I.append(torch.argmax(d[v])) 
    
    E = Yp[:, I] 
        
    if E_gt is not None: 
        E_ordered, E_ordered_norm, indices = utils.order_endmembers(E, E_gt) 
        return E_ordered 
    else: 
        return E

def VCA(Y, c, snr_input=0):
    """
    Vertex Component Analysis algorithm by Jose M. P. Nascimento and Jose M. B. Dias
    
    Args:
        Y: input HSI to extract endmembers from (shape (B, h, w) or (B, N))
        c (int): the number of endmembders to extract
        snr_input: the snr of the input image (default: 0)
    """
    
    if Y.dim()!= 2:
        if Y.dim() == 4:
            Y = Y[0]
        B, h, w = Y.shape
        N = h*w
        Y = Y.reshape(B, N)
    else:
        B, N = Y.shape
    
    if snr_input == 0:
        y_m = torch.mean(Y, dim=1, keepdim=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = LA.svd(torch.matmul(Y_o, Y_o.T) / float(N))[0][:, :c]  # computes the R-projection matrix
        x_c = torch.matmul(Ud.T, Y_o)  # project the zero-mean data onto c-subspace

        SNR = estimate_snr(Y, y_m, x_c)
    else:
        SNR = snr_input

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

    return E

def batched_VCA(Y, c, seed=None, snr_input=0, verbose=False):
    """
    Vertex Component Analysis algorithm by Jose M. P. Nascimento and Jose M. B. Dias
    
    Args:
        Y: input HSI to extract endmembers from (shape (batch, B, h, w) or (batch, B, N))
        c (int): the number of endmembders to extract
        snr_input: the snr of the input image (default: 0)
        verbose (bool, optional): whether to display informations or not (default: False)
    """
    
    if Y.dim() == 4:  # (batch, B, H, W)
        batch, B, H, W = Y.shape
        N = H * W
        Y = Y.reshape(batch, B, N)
    elif Y.dim() == 3:  # (batch, B, N)
        batch, B, N = Y.shape
    else:
        raise ValueError("Y must be 3D or 4D tensor")

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = torch.Generator()
        
    if snr_input == 0:
        y_m = torch.mean(Y, dim=2, keepdim=True)
        Y_o = Y - y_m  # data with zero-mean
        Y_o_Y_oT = torch.matmul(Y_o, Y_o.transpose(1, 2)) / float(N)  # (batch, B, B)

        # Compute SVD for all batches
        U, S, Vh = LA.svd(Y_o_Y_oT)  # U: (batch, B, B), S: (batch, B), Vh: (batch, B, B)
        Ud = U[:, :, :c]  # (batch, B, c)

        # Project zero-mean data onto c-subspace for all batches
        x_c = torch.matmul(Ud.transpose(1, 2), Y_o)  # (batch, c, N)

        SNR = batched_estimate_snr(Y, y_m, x_c)

        if verbose:
            print(f"input SNR estimated = {SNR}[dB]")
    else:
        SNR = snr_input
        
        if verbose:
            print(f"input SNR = {SNR}[dB]\n")

    # Compute SNR threshold
    SNR_th = 15 + 10 * torch.log10(torch.tensor(c))

    if SNR < SNR_th:

        d = c - 1
        if snr_input == 0:  # it means that the projection is already computed
            Ud = Ud[:, :d]
        else:
            y_m = torch.mean(Y, dim=2, keepdim=True)
            Y_o = Y - y_m  # data with zero-mean

            Ud = LA.svd(torch.matmul(Y_o, Y_o.transpose(1,2)) / float(N))[0][:, :, :d]  # computes the c-projection matrix
            x_c = torch.matmul(Ud.transpose(1,2), Y_o)  # project thezeros mean data onto c-subspace

        Yc = torch.matmul(Ud, x_c[:, :d, :]) + y_m  # again in dimension c

        x = x_c[:, :d, :]  #  x_c =  Ud.T * Y_o is on a R-dim subspace
        b = torch.max(torch.sum(x**2, dim=0)) ** 0.5
        y = torch.vstack((x, b * torch.ones((1, N))))
    else:

        d = c
        Ud = LA.svd(torch.matmul(Y, Y.transpose(1,2)) / float(N))[0][:, :, :c]  # computes the c-projection matrix

        x_c = torch.matmul(Ud.transpose(1,2), Y)
        Yc = torch.matmul(Ud, x_c)  # again in dimension b (note that x_c has no null mean)

        x = torch.matmul(Ud.transpose(1,2), Y)
        u = torch.mean(x, dim=2, keepdim=True)  # equivalent to  u = Ud.T * r_m
        y = x / torch.matmul(u.transpose(1,2), x)
        
    
    indices = torch.zeros((batch, c), dtype=torch.long)
    A = torch.zeros((batch, c, c))
    A[:, -1, 0] = 1

    # Iterate c times (vectorized)
    for i in range(c):
        # Random projection for all batches
        w = torch.rand((batch, c, 1), device=Y.device)
        f = w - torch.matmul(A, torch.matmul(LA.pinv(A), w))  # (batch, c, 1)
        f = f / LA.norm(f, dim=1, keepdim=True)  # (batch, c, 1)

        v = torch.matmul(f.transpose(1, 2), y).squeeze(1)  # (batch, N)
        
        indices[:, i] = torch.argmax(torch.abs(v), dim=1)  # (batch,)
        A[:, :, i] = y[torch.arange(batch), :, indices[:, i]]  # (batch, c)

    # Gather E for all batches
    E = Yc[torch.arange(batch), :, indices].transpose(1,2)  # (batch, B, c)

    
    if verbose:
        print(f"Indices chosen to be the most pure: {indices}")
    return E  # (batch, B, c)   
    
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
    # abd_fcls = abd_ls

    # Clip negative abundances to zero
    abd_clipped = torch.clamp(abd_fcls, min=0)
    sum_abd = torch.sum(abd_clipped, dim=0, keepdim=True)  # (1 x N)
    
    # Avoid division by zero (if all abundances are zero for a pixel)
    sum_abd[sum_abd == 0] = 1  # Set to 1 to avoid NaN
    
    A = abd_clipped / sum_abd

    return A

def project_rows_to_simplex(H):
    # H: (n_rows, dim)
    sorted_H, _ = torch.sort(H, descending=True, dim=1)
    cumsum = torch.cumsum(sorted_H, dim=1) - 1
    k = (sorted_H > (cumsum / torch.arange(1, H.size(1)+1, device=H.device))).sum(dim=1) - 1
    theta = cumsum[torch.arange(H.size(0)), k] / (k + 1)
    return torch.clamp(H - theta.unsqueeze(1), min=0)

def FCLS_2(Y, E, A0=None, inner_iter=300, delta=1e-6, alpha0=0.05):
    """
    Solves min_H >= 0  ||Y - EA||_F^2 using fast projected gradient.

    Y : (B, N)
    E : (B, c)
    A0: optional (c, N)
    """

    # Precompute constants
    EtE = E.T @ E           # c x c
    EtY = E.T @ Y           # c x N
    L = torch.linalg.norm(EtE, 2)  # Lipschitz

    # Initialization
    if A0 is None:
        A = torch.clamp(EtY / (torch.diag(EtE).unsqueeze(1) + 1e-12), min=0)
    else:
        A = A0.clone()

    # Choose projection function
    project = lambda Z: project_rows_to_simplex(Z.T).T # keep A as c×N

    A = project(A)
    A_ = A.clone()

    # FPGM parameters
    alpha_prev = alpha0

    eps0 = None
    for _ in range(inner_iter):

        A_prev = A
        # Gradient: Wᵀ(WA_ - X)
        grad = EtE @ A_ - EtY

        # Forward step
        A = A_ - grad / L

        # Projection
        A = project(A)

        # Compute FPGM update
        alpha_new = (torch.sqrt(torch.tensor(alpha_prev**4 + 4*alpha_prev**2)) - alpha_prev**2) / 2
        beta = alpha_prev * (1 - alpha_prev) / (alpha_prev**2 + alpha_new)

        # Nesterov update
        Y = A + beta * (A - A_prev)

        # Stopping criterion
        diff = torch.norm(A - A_prev, p='fro')
        if eps0 is None:
            eps0 = diff
        if diff <= delta * eps0:
            break

        alpha_prev = alpha_new

    return A

def unmix(y, c, use_sivm=False):
    """
    Unmixes the input HSI y into c endmembers by applying VCA + FCLS
    
    Args:
        y (torch.tensor): the input hsi tensor (shape (B, h, w) or (B, N))
        c (int): the number of endmembers to unmix
    """
    
    if use_sivm:
        E = SiVM(y, c)
    else:
        E = VCA(y, c)
    A = FCLS(y, E)
    
    return E, A
    