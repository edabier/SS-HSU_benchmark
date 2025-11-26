import torch
import torch.nn as nn
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import src.training.data_augmentation as data_aug
import src.utils.extractor as extractor
import src.utils.utils as utils

class SelfSupervisedTrainer():
    def __init__(self):
        pass
    
    def train(self, y):
        raise NotImplementedError(f"Training method is not implemented for {self}")

class DIP(SelfSupervisedTrainer):
    """
    Defines a Deep Image Prior-type of training based on Ulyanov et al. 2020.
    We optimize the model to reconstruct an input image y from random gaussian noise:
    
    y* = min_f || y_gt - f(z) ||

    Args:
        model: the model to train
        criterion: the function to optimize by training the model, by default the MSE loss (default: None)
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    def __init__(self, model, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.MSELoss()
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            z = torch.randn_like(y)
            e_hat, a_hat, y_hat = self.model(z)
            
            loss = self.criterion(y, y_hat)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return e_hat, a_hat, train_losses
    
class TwoStagesNet(SelfSupervisedTrainer):
    """
    Defines a Two stages Net-type of training based on Vijayashekhar et al.2022
    We optimize the model to reconstruct an input image y and force it to be a good denoiser at the same time
    We create a small MLP that is trained to denoise the output of the model:
    
    y -> model(y) = r -> r+n -> MLP(r+n) -> y_hat
    
    We train the entire model (input model + MLP)

    Args:
        model: the model to train
        B (int): the number of spectral bands of the input image
        criterion: the function to optimize by training the model, by default, we use the loss defined in the article (default: None)
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    def __init__(self, model, B, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            
        if criterion is not None:
            self.criterion = criterion
            
        self.denoiser = nn.Sequential(
            nn.Linear(B, 120), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(120, 90), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(90, 45), nn.ReLU(), nn.Dropout(p=0.3), 
            nn.Linear(45, B))
    
    def criterion(y_gt, y_hat, r, n):
        """
        The loss is the sum of:
        - MSE(y_hat, y_gt)
        - MSE(r+n, y_gt)
        - SAD(r+n, y_gt)
        """
        mse = nn.MSELoss()
        sad = utils.SADLoss()
        
        loss_forward = mse(y_gt, y_hat)
        loss_denoiser = mse(y_gt, (r+n))
        loss_sad = sad(y_gt, (r+n))
        
        return loss_forward + loss_denoiser + loss_sad
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            e_hat, a_hat, r = self.model(y)
            n = torch.randn_like(y)
            r += n
            y_hat = self.denoiser(r)
            
            loss = self.criterion(y, y_hat, r, n)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return e_hat, a_hat, train_losses
    
class ReconstructionError(SelfSupervisedTrainer):
    """
    Defines a training based simply on the reconstruction error

    Args:
        model: the model to train
        criterion: the function to optimize by training the model, by default the MSE loss (default: None)
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    def __init__(self, model, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.MSELoss()
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            e_hat, a_hat, y_hat = self.model(y)
            
            loss = self.criterion(y, y_hat)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return e_hat, a_hat, train_losses
        
    
class EGU_Net(SelfSupervisedTrainer):
    """
    Defines an Endmember Guided SSL training method based on Hong et al. 2021
    We optimize the model to move the representation of similar patches close together, and move apart different ones
    We use the NT-Xent loss for this
    
    We split the input image Y in patches
    We create positive augmentations of each patch
    We select negative patches in the image that don't contain the same EMs as the current patch
    We forward the patch in the model to get estimated A and E
    We project the estimated As of the positive and negative samples using the projection head
    We compute the NT-Xent loss between every pair, minimizing it for positive pairs, and maximizing it for negative ones

    Args:
        model: the model to train
        c (int): the number of endmembers to extract from the input HSI
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    def __init__(self, model, c, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.c = c
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
    def extract_endmember_bundle(self, y, sub_size=0.1, n_sub=20, replacement=False):
        """
        Uses Somers et al. 2012 method to extract bundles of endmembers and their abundances
        
        We start by randomly sampling 20 sub-images of size 10% of the original size (without replacement)
        Then we apply VCA to each subset to extract the pure pixels, and create a library with them
        We use k-means to cluster the ems in bundles
        We randomly select ems from each bundles to find the abundances of each pure pixels using FCLS
        
        Args:
            y: the input HSI in which to extract ems
            sub_size (float, optional): the size of the sub images to sample (default: 0.1)
            n_sub (int, optional): the number of sub images to sample (default: 20)
            replacement (bool, optional): whether to sample with replacement or not (default: False)
        """
        
        sub_h = int(sub_size * y.shape[1])
        sub_w = int(sub_size * y.shape[2])
        
        # Calculate the maximum possible top-left corner indices
        # Generate all possible top-left corner indices
        max_i = y.shape[1] - sub_h
        max_j = y.shape[2] - sub_w
        all_i = torch.arange(0, max_i + 1)
        all_j = torch.arange(0, max_j + 1)
        all_ij = torch.cartesian_prod(all_i, all_j)
        
        if replacement:
            sampled_indices = torch.randint(0, len(all_ij), (n_sub,))
        else:
            if n_sub > len(all_ij):
                raise ValueError(f"EGU-Net extract bundles: Cannot sample {n_sub} unique sub-images. Maximum possible is {len(all_ij)}.")
            sampled_indices = torch.randperm(len(all_ij))[:n_sub]

        sampled_ij = all_ij[sampled_indices]
        sub_images = torch.stack([y[:, i:i+sub_h, j:j+sub_w] for i, j in sampled_ij])
        
        em_lib = torch.stack([extractor.VCA(sub_images[i], self.c) for i in range(n_sub)], dim=1) # shape (B, c * n_sub)
        
        centers, memberships = data_aug.group_spectra_kmeans(em_lib.T, n_clusters=self.c)
        grouped_lib = data_aug.group_spectra_by_cluster(em_lib, memberships) # shape (c, B, nb_spectra_in_cluster)
        e_avg = torch.stack([torch.mean(grouped_lib[i], dim=1, keepdim=True) for i in range(c)], dim=1).squeeze(2)
        
        # Apply FCLS to randomly sampled ems in grouped_lib 
        E = torch.stack([grouped_lib[i][:,random.randint(0, grouped_lib[i].shape[1]-1)] for i in range(len(grouped_lib))])
        pure_abds = extractor.FCLS(y, E)
        
        return pure_abds, e_avg
        
    def train(self, y):
        
        train_losses = []
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()
        
        for epoch in range(self.epochs):
        
            self.optimizer.zero_grad()
            
            # Top part: unmixing pure pixels
            # We create a small 2x2 image by repeating each pure pixel 4 times, and forward it to the model's encoder
            abd, e_avg = self.extract_endmember_bundle(y)
            end_mat = [torch.zeros(0)]*self.c
            end_mat = [torch.cat((end_mat[i], e_avg[:,i].repeat(4))).reshape(e_avg.shape[0], 2, 2).unsqueeze(0) for i in range(self.c)]
            predicted_abd = [self.model.encoder(end_mat[i]) for i in range(self.c)]
            
            # Cross entropy between abd and unmixed ems (since we repeated em to make it 2d, we only take the first dim)
            loss = torch.sum([ce(abd[i], predicted_abd[i].squeeze(0)[:,0,0]) for i in range(self.c)])
            
            e_hat, a_hat, y_hat = self.model(y)
            
            # MSE between y_hat and y   
            loss += mse(y, y_hat)
            train_losses.append(loss)
                
            loss.backward()
            self.optimizer.step()
        
        return e_hat, a_hat, train_losses
    
class GeneratedDataset(SelfSupervisedTrainer):
    """
    Uses the input HSI to generate an extended dataset based on Hadjeres et al. 2024
    
    Args:
        model: the model to be trained
        dataset_size (int, optional): the number of tuple (Yi, Ei, Ai) to generate (default: 10000)
        criterion: the function to optimize by training the model, by default, we use the loss defined in the article (default: None)
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    
    def __init__(self, model, dataset_size=10000, criterion=None, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            
        if criterion is not None:
            self.criterion = criterion
    
    def create_dataset(self, y, c=4, n_vca=10, n_aug=10, c_var=0.4):
        """
        - We first run n times the VCA algorithm to extract different EM
        - We group them and remove duplicate with a K-means algorithm to construct a library
        - We augment this library by generating variations of each material with piece-wise affine functions
        - We select the average spectra of each material of the augmented library to create the EM matrix
        - We apply FCLS to this EM matrix and the y HSI to obtain an estimation of abundance map
        - We estimate the parameters of a Dirichlet mixture to model the abundance distribution of every material
        To do this, we use an Expectation-Maximization algorithm, and we find the number of modes with the AIC 
        
        => What does this distribution represents? The pixel distribution of the materials on the image?
        
        - We create a new point in the dataset by randomly picking:
            - an EM matrix from the library E
            - an abundance map following the estimate Dirichlet mixture distribution A
            - a mixed image y obtained by: y = EA + N (adding noise to have SNR=30dB)
        
        Args:
            y: the input HSI to unmix and with which to create a dataset
            c (int, optional): the number of endmembers to extract (default: 4)
            n_vca (int, optional): the number of times to run the VCA (default: 10)
            n_aug (int, optional): the number of variations of each spectra of the library to create (default: 10)
            c_var (float, optional): the variability coefficient (default: 0.4)
        """
        
        B, h, w = y.shape
        
        self.dataset = {"E": [], "A": [], "Y": []}
        
        endmember_lib = torch.tensor([])

        # We run n_vca times the VCA extraction to get n_vca*c endmembers
        for _ in range(n_vca):
            e = extractor.VCA(y, c=c) # shape (B, c)
            endmember_lib = torch.cat((endmember_lib, e), dim=1)

        # We remove duplicate ems and normalize them
        unique_spectra = data_aug.remove_duplicates(endmember_lib, tol=1e-4)
        norms = torch.linalg.norm(unique_spectra, dim=1, keepdim=True)
        unique_spectra_norm = unique_spectra / norms # shape (B, c*nb_spectra_in_cluster)

        # We use Kmeans to cluster the ems to create exactly c categories
        centers, memberships = data_aug.group_spectra_kmeans(unique_spectra_norm.T, n_clusters=c)
        grouped_lib = data_aug.group_spectra_by_cluster(unique_spectra_norm, memberships) # shape (c, B, nb_spectra_in_cluster)

        # We augment the number of ems in each cluster by running n_aug times the augmentation function
        # Augmented_lib has shape (c, B, n_aug*nb_spectra_in_cluster)
        augmented_lib = [
            torch.cat([torch.stack([data_aug.augment_spectrum(group[:, i], c_var) for _ in range(n_aug)], dim=1)
                    for i in range(group.shape[1])], dim=1)
        for group in grouped_lib]
        
        # We average the ems of each cluster to find an average E matrix
        # e_avg has shape (B, c)
        e_avg = torch.stack([torch.mean(augmented_lib[i], dim=1, keepdim=True) for i in range(c)], dim=1).squeeze(2)
        
        # Apply FCLS on y with the average endmembers to obtain the "average" abundance matrices
        a_avg = extractor.FCLS(y, e_avg)
        # TO DO: a_model = DirichletMixtureModel(a_avg)
        
        for i in range(self.dataset_size):
            
            # TO DO: 
            # Sample Ai from the learned Dirichlet Mixture distribution
            # Ai = a_model()
            
            # Sample Ei from the augmented endmembers library
            Ei = torch.stack([augmented_lib[i][:,random.randint(0, augmented_lib[i].shape[1])] for i in range(c)]).T
            # Yi = Ei @ Ai
            
            self.dataset.E.append(Ei)
            self.dataset.A.append(i)
            self.dataset.Y.append(i)
    
    def criterion(self, e_gt, e_hat, a_gt, a_hat):
        """
        Computes the loss between the predicted and target E and A
        """
        
        mse = nn.MSELoss()
        sad = utils.SADLoss()
        
        loss_e = sad(e_gt, e_hat)
        loss_a = mse(a_gt, a_hat)**0.5
        
        return loss_e + loss_a
    
    def train(self, y):
        
        # Make sure the synthetic training dataset has been created first
        assert hasattr(self, "dataset"), "The training dataset must be generated first by running self.create_datatset()"
        
        train_losses = []
        for _ in range(self.epochs):
            
            for i in range(self.dataset_size):
                
                e_gt = self.dataset.E[i]
                a_gt = self.dataset.A[i]
                y_gt = self.dataset.Y[i]
                
                self.optimizer.zero_grad()
                
                e_hat, a_hat, y_hat = self.model(y_gt)
                
                loss = self.criterion(e_gt, e_hat, a_gt, a_hat)
                train_losses.append(loss)
                
                loss.backward()
                self.optimizer.step()
            
        return e_hat, a_hat, train_losses

class ContrastiveLearning(SelfSupervisedTrainer):
    """
    Defines a contrastive training method based on Zhao et al.2022
    We optimize the model to move the representation of similar patches close together, and move apart different ones
    We use the NT-Xent loss for this
    
    We split the input image Y in patches
    We create positive augmentations of each patch
    We select negative patches in the image that don't contain the same EMs as the current patch
    We forward the patch in the model to get estimated A and E
    We project the estimated As of the positive and negative samples using the projection head
    We compute the NT-Xent loss between every pair, minimizing it for positive pairs, and maximizing it for negative ones

    Args:
        model: the model to train
        projection_head: the small model used to project the abundances map on a space on which we compute the loss
        optimizer: the optimizer to use for the training, by default, we use AdamW (default: None)
    """
    def __init__(self, model, projection_head, optimizer=None, epochs=200, lr=0.001, batch_size=1):
        super().__init__()
        self.model = model
        self.projection_head = projection_head
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    
    def cosine_sim(self, A, B):
        """
        Computes the cosine similarity between two matrices
        """
        num = A.T @ B
        denom = torch.norm(A) * torch.norm(B)
        return num/ denom
    
    def criterion(self, a, a_positive, a_negative, temp=0.5):
        """
        Defines the NT-Xent loss
        
        Args:
            a: the augmented A matrix
            a_positive (list): the list of positive pairs generated from a
            a_negative (list): the list of negative pairs for a
            temp (float): the temperature parameter (default: 0.5)
        """
        loss = 0
        sad = utils.SADLoss()
        
        # for a_pos in a_positive:
        #     num = torch.exp(self.cosine_sim(a, a_pos))/ temp
        #     denom = torch.sum([torch.exp(self.cosine_sim(a, a_negative[i]))/ temp for i in range(len(a_negative))])
        #     l = - torch.log(num/denom)
        #     loss += l
        
        for a_pos in a_positive:
            num = torch.exp(sad(a.T, a_pos.T))/ temp
            denom = torch.sum([torch.exp(sad(a.T, a_negative[i].T))/ temp for i in range(len(a_negative))])
            l = - torch.log(num/denom)
            loss += l
            
        return loss
    
    def find_negative_patches(self, y, c, n_patches, patch_size, patch_pos, average=False):
        """
        Finds patches in y where there are different materials than in y_patch
        We do this by estimating the endmembers and their abundance in y (VCA + FCLS)
        Then, we find patches in y with the most different composition from the patch at patch_pos using the cosine similiarity 
        Returns the top left coordinates of the n_patches with the most dissimilarity from the current patch
        
        Args:
            y: the input HSI image (shape (B, h, w) or (B, N))
            n_patches (int): the amount of negative patches to find
            patch_size (int): the size of patches to split the y HSI
            patch_pos: the position of the "positive" patch to compare to negative patches
            average (bool, False): whether or not to average the abundances of the patches for easier comparison (default: False)
        Returns:
            neg_patch_coords (list): the list of the top left coordinates 
        """
        
        # Reshape input HSI to a cube
        if y.dim() == 2:
            y = y.reshape(y.shape[0], int(y.shape[1]**0.5), int(y.shape[1]**0.5))
        B, H, W = y.shape
        
        E, A = extractor.unmix(y, c)
        A = A.reshape(c, int(A.shape[1]**0.5), int(A.shape[1]**0.5))
        
        x, y = patch_pos
        current_patch = A[:, x:x+patch_size, y:y+patch_size]
        
        if average:
            # Average the abundances over the patch
            current_abundance = current_patch.reshape(c, -1).mean(axis=1)
        else:
            # Use the entire patch as a flattened vector
            current_abundance = current_patch.reshape(-1)
        
        patches_unfold = nn.functional.unfold(A, kernel_size=patch_size, stride=1)
        patches_unfold = patches_unfold.permute(1, 0)  # (num_patches, c * patch_size**2)

        # Reshape to (num_patches, c, patch_size**2) if not averaging
        if not average:
            patches_unfold = patches_unfold.view(-1, c, patch_size**2)

        # Compute patch abundances
        if average:
            patches = patches_unfold.view(-1, c, patch_size**2).mean(dim=2)  # (num_patches, c)
        else:
            patches = patches_unfold.view(-1, c * patch_size**2)  # (num_patches, c * patch_size**2)

        # Compute similarity
        similarities = cosine_similarity(patches, current_abundance.reshape(1, -1)).flatten()

        _, least_similar_indices = torch.topk(torch.tensor(similarities), k=n_patches, largest=False)

        # Generate patch coordinates
        patch_coords = [(i // (W - patch_size + 1), i % (W - patch_size + 1)) for i in least_similar_indices]

        return patch_coords
    
    def create_positive_patches(self, y, crop, flip, blur, spectral, n_pairs):
        """
        Creates n_pairs positive pairs of the input HSI y
        
        Args:
            y: input HSI to be augmented
            crop (float): the probability with which to apply cropping
            flip (float): the probability with which to apply flipping
            blur (float): the probability with which to apply  blurring
            spectral (float): the probability with which to apply spectral variation
            n_pairs (int): the number of pairs to generate
        """
        positive_pairs = []
        
        for _ in range(n_pairs):
            rand_crop = torch.rand(1)
            if rand_crop < crop:
                aug_y = data_aug.crop_and_resize(y, r=0.95)
                
            rand_flip = torch.rand(1)
            if rand_flip < flip:
                aug_y = data_aug.flip(y)
            
            rand_blur = torch.rand(1)
            if rand_blur < blur:
                aug_y = data_aug.blur(y, r=3, sigma=2)
            
            rand_spectral = torch.rand(1)
            if rand_spectral < spectral:
                aug_y = data_aug.spectral_variability(y, c_var=0.4)
            
            try:
                positive_pairs.append(aug_y)
            except:
                print("No augmentation applied, returning y")
                return y
        
        return positive_pairs
    
    def train(self, y):
        train_losses = []
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            
            y_positives = self.create_positive_patches(y, crop=1, flip=0, blur=0, spectral=0.8, n_pairs=2)
            y_negative = self.find_negative_patches(y)
            
            e_hat, a_hat, x_hat = self.model(y_positives[0])
            e_hat_pos, a_positive, x_hat_pos = self.model(y_positives[1])
            e_hat_neg, a_negative, x_hat_neg = self.model(y_negative)
            
            a_projected = self.projection_head(a_hat)
            a_positive_projected = self.projection_head(a_positive)
            a_negative_projected = self.projection_head(a_negative)
            
            loss = self.criterion(a_projected, a_positive_projected, a_negative_projected)
            train_losses.append(loss)
            
            loss.backward()
            self.optimizer.step()
            
        return e_hat, a_hat, train_losses