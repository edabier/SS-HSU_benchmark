import torch
import torch.nn as nn
import data_augmentation as data_aug
import utils.extractor as extractor
import utils.utils as utils

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
        
        vca = extractor.VCA()
        endmember_lib = torch.tensor([])

        # We run n_vca times the VCA extraction to get n_vca*c endmembers
        for _ in range(n_vca):
            e = vca.extract_endmembers(y, c=c) # shape (B, c)
            endmember_lib = torch.cat((endmember_lib, e), dim=1)

        # We remove duplicate ems and normalize them
        unique_spectra = data_aug.remove_duplicates(endmember_lib, tol=1e-4)
        norms = torch.linalg.norm(unique_spectra, dim=1, keepdim=True)
        unique_spectra_norm = unique_spectra / norms # shape (B, c*nb_spectra_in_cluster)

        # We use Kmeans to cluster the ems to create exactly c categories
        centers, memberships = data_aug.group_spectra_kmeans(unique_spectra_norm.T, n_clusters=c)
        grouped_lib = data_aug.group_spectra_by_cluster(unique_spectra_norm, memberships) # shape (c, N, nb_spectra_in_cluster)

        # We augment the number of ems in each cluster by running n_aug times the augmentation function
        # Augmented_lib has shape (c, B, n_aug*nb_spectra_in_cluster)
        augmented_lib = [
            torch.cat([torch.stack([data_aug.augment_spectrum(group[:, i], c_var) for _ in range(n_aug)], dim=1)
                    for i in range(group.shape[1])], dim=1)
        for group in grouped_lib]
        
        # We average the ems of each cluster to find an average E matrix
        # e_avg has shape (B, c)
        e_avg = torch.stack([torch.mean(augmented_lib[i], dim=1, keepdim=True) for i in range(c)], dim=1).squeeze(2)
        
        for i in range(self.dataset_size):
            
            # TO DO: implement/ retrieve dataset creation code
            
            self.dataset.E.append(i)
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
        
        for a_pos in a_positive:
            num = torch.exp(self.cosine_sim(a, a_pos))/ temp
            denom = torch.sum([torch.exp(self.cosine_sim(a, a_negative[i]))/ temp for i in range(len(a_negative))])
            l = - torch.log(num/denom)
            loss += l
            
        return loss
    
    def find_negative_patches(self, y, patch_size, patch_pos):
        """
        Finds patches in y where there are different materials than in y_patch
        We do this by estimating the endmembers and their abundance in y (VCA + FCLS)
        Then, we find patches in y with the most different composition from the patch at patch_pos
        
        Args:
            y: input HSI image
            patch_size: The size of patches to split the y HSI
            patch_pos: the position of the "positive" patch to compare to negative patches
        """
        pass
    
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