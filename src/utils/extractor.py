import torch
import torch.linalg as LA
from sklearn.cluster import KMeans
import numpy as np
import time

class VCA():
    """
    VCA algorithm, from HySUPP, adapted to torch
    """
    def __init__(self, seed=None):
        """
        Args:
            seed (int, optional): random seed for the algorithm (default: None)
        """
        self.seed = seed

    def extract_endmembers(self, Y, c, snr_input=0):
        """
        Vertex Component Analysis algorithm by Jose M. P. Nascimento and Jose M. B. Dias
        
        Args:
            Y: input HSI to extract endmembers from (shape (b, h, w))
            c (int): the number of endmembders to extract
            snr_input: the snr of the input image (default: 0)
        """
        b, h, w = Y.shape
        N = h*w
        Y = Y.reshape(b, N)
        
        if self.seed is not None:
            generator = torch.Generator().manual_seed(self.seed)
        else:
            generator = torch.Generator()
        
        if snr_input == 0:
            y_m = torch.mean(Y, dim=1, keepdim=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = LA.svd(torch.matmul(Y_o, Y_o.T) / float(N))[0][:, :c]  # computes the R-projection matrix
            x_c = torch.matmul(Ud.T, Y_o)  # project the zero-mean data onto c-subspace

            SNR = self.estimate_snr(Y, y_m, x_c)

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
        self.indices = indices

        return E

    @staticmethod
    def estimate_snr(Y, r_m, x):
        N, c = Y.shape # c number of bands (channels), N number of pixels
        p, N = x.shape  # p number of endmembers (reduced dimension)
        P_y = torch.sum(Y**2) / float(N)
        P_x = torch.sum(x**2) / float(N) + torch.sum(r_m**2)
        snr_est = 10 * torch.log10((P_x - p / c * P_y) / (P_y - P_x))

        return snr_est
    
    # def remove_duplicates(self, lib, tol=1e-3):
    #     """
    #     Removes duplicates from spectra library
    #     """
    #     unique_spectra = []
        
    #     for spec in lib.T:
    #         if not any(torch.linalg.norm(spec - uniq) < tol for uniq in unique_spectra):
    #             unique_spectra.append(spec)
    #     return torch.stack(unique_spectra).T
    
    # def group_spectra_kmeans(self, spectra, n_clusters, seed=42):
    #     """
    #     Groups spectra using scikit-learn's Kmeans algorithm.
    #     """
    #     spectra = spectra.numpy()
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    #     labels = kmeans.fit_predict(spectra)
    #     centers = kmeans.cluster_centers_
        
    #     return centers, labels
    
    # def group_spectra_by_cluster(self, lib, memberships):
    #     """
    #     Groups spectra by cluster by attributing the cluster by maximum belonging degree.
        
    #     Args:
    #         lib: spectra library of shape (n_spectra, c)
    #         memberships: belonging matrix of shape (n_clusters, n_spectra)
        
    #     Returns:
    #         groups: groups[i] contains cluster's i spectras of shape (nb_spectra_in_cluster, c).
    #     """
    #     # Attribution to cluster using argmax on the cluster's dimension
    #     n_clusters = memberships.max().item() + 1
    #     groups = []
    #     for i in range(n_clusters):
    #         indices = np.where(memberships == i)[0]
    #         group = torch.stack([lib[:, idx] for idx in indices], dim=1)
    #         groups.append(group)
    #     return groups
    
    # def augment_spectrum(self, spectrum, c_var=0.4):
    #     """
    #     Applies a randomly drawn piece-wise affine function to the tensor's bands
        
    #     Args:
    #         spectrum (torch.tensor): the input spectrum to be transformed
    #         c_var (float): the variability coefficient
    #     """
    #     c = len(spectrum)
    #     # Randomly draw the piece-wise affine function points
    #     y0 = torch.rand(1) * c_var + (1 - c_var/2)
    #     y1 = torch.rand(1) * c_var + (1 - c_var/2)
    #     y2 = torch.rand(1) * c_var + (1 - c_var/2)
    #     x = torch.floor(c/2 + torch.floor(c * torch.randn(1) / 3)).clamp(0, c)
        
    #     # Create a tensor of band indices (t) for each pixel
    #     t = torch.arange(c).float()
        
    #     return spectrum * torch.where(
    #         t <= x,
    #         y0 + (y1 - y0) * t / x,
    #         y1 + (y2 - y1) * (t - x) / (c - x)
    #     )

# class FCLS(SupervisedUnmixingModel):
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     @staticmethod
#     def _numpy_None_vstack(A1, A2):
#         if A1 is None:
#             return A2
#         else:
#             return np.vstack([A1, A2])

#     @staticmethod
#     def _numpy_None_concatenate(A1, A2):
#         if A1 is None:
#             return A2
#         else:
#             return np.concatenate([A1, A2])

#     @staticmethod
#     def _numpy_to_cvxopt_matrix(A):
#         A = np.array(A, dtype=np.float64)
#         if A.ndim == 1:
#             return matrix(A, (A.shape[0], 1), "d")
#         else:
#             return matrix(A, A.shape, "d")

#     def compute_abundances(self, Y, E, *args, **kwargs):
#         """
#         Performs fully constrained least squares of each pixel in M
#         using the endmember signatures of U. Fully constrained least squares
#         is least squares with the abundance sum-to-one constraint (ASC) and the
#         abundance nonnegative constraint (ANC).
#         Parameters:
#             Y: `numpy array`
#                 2D data matrix (L x N).
#             E: `numpy array`
#                 2D matrix of endmembers (L x p).
#         Returns:
#             X: `numpy array`
#                 2D abundance maps (p x N).
#         References:
#             Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
#             Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.
#         Notes:
#             Three sources have been useful to build the algorithm:
#                 * The function hyperFclsMatlab, part of the Matlab Hyperspectral
#                 Toolbox of Isaac Gerg.
#                 * The Matlab (tm) help on lsqlin.
#                 * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
#                 http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
#                 , it's great code.
#         """
#         tic = time.time()
#         assert len(Y.shape) == 2
#         assert len(E.shape) == 2

#         L1, N = Y.shape
#         L2, p = E.shape

#         assert L1 == L2

#         # Reshape to match implementation
#         M = np.copy(Y.T)
#         U = np.copy(E.T)

#         solvers.options["show_progress"] = False

#         U = U.astype(np.double)

#         C = self._numpy_to_cvxopt_matrix(U.T)
#         Q = C.T * C

#         lb_A = -np.eye(p)
#         lb = np.repeat(0, p)
#         A = self._numpy_None_vstack(None, lb_A)
#         b = self._numpy_None_concatenate(None, -lb)
#         A = self._numpy_to_cvxopt_matrix(A)
#         b = self._numpy_to_cvxopt_matrix(b)

#         Aeq = self._numpy_to_cvxopt_matrix(np.ones((1, p)))
#         beq = self._numpy_to_cvxopt_matrix(np.ones(1))

#         M = np.array(M, dtype=np.float64)
#         M = M.astype(np.double)
#         X = np.zeros((N, p), dtype=np.float32)
#         for n1 in range(N):
#             d = matrix(M[n1], (L1, 1), "d")
#             q = -d.T * C
#             sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)["x"]
#             X[n1] = np.array(sol).squeeze()

#         # Record time
#         self.time = time.time() - tic

#         return X.T
