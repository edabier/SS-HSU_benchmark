import numpy as np
import matplotlib.pyplot as plt
from scipy import io as sio
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def augment_spectrum(spectrum, cvar=0.4):
    """
    Applique une fonction affine par morceaux au spectre pour simuler la variabilité.
    Trois coefficients de contrôle ξ1, ξ2, ξ3 sont tirés aléatoirement selon U[1-cvar/2, 1+cvar/2].
    On définit un point de rupture Lbreak dans [1, L-2] et on interpole linéairement
    entre ces trois valeurs pour obtenir la fonction d'augmentation.
    """
    L = len(spectrum)
    # Tirage des coefficients aléatoires
    xi1 = np.random.uniform(1 - cvar/2, 1 + cvar/2)
    xi2 = np.random.uniform(1 - cvar/2, 1 + cvar/2)
    xi3 = np.random.uniform(1 - cvar/2, 1 + cvar/2)
    # Tirage d'une valeur U ~ N(0,1) et calcul du point de rupture
    U = np.random.randn()
    Lbreak = int(np.clip(np.floor(L/2) + np.floor(L * abs(U)/3), 1, L-2))
    
    # Construction de la fonction d'augmentation par interpolation linéaire
    f = np.empty(L)
    for i in range(L):
        if i <= Lbreak:
            # Interpolation entre xi1 et xi2
            f[i] = xi1 + (xi2 - xi1) * (i / Lbreak)
        else:
            # Interpolation entre xi2 et xi3
            f[i] = xi2 + (xi3 - xi2) * ((i - Lbreak) / (L - Lbreak - 1))
    
    return spectrum * f

def remove_duplicates(lib, tol=1e-3):
    """
    Removes duplicates from spectra library
    
    Args :
        spectra : (n_spectra, c) shaped tensor with c the number of bands.
        tol : tolerance threshold to consider two similar spectra.
    
    Returns :
        unique_spectra : library containing unic spectra.
    """
    unique_spectra = []
    for spec in lib:
        if not any(np.linalg.norm(spec - uniq) < tol for uniq in unique_spectra):
            unique_spectra.append(spec)
    return np.array(unique_spectra)

def group_spectra_kmeans(spectra, n_clusters, seed=42):
    # """
    # Regroupe les spectres en utilisant l'algorithme fuzzy c-means de scikit-fuzzy.
    
    # Paramètres :
    #   - spectra : array de forme (n_spectra, L)
    #   - n_clusters : nombre de clusters (matériaux attendus)
    #   - m : exponent de fuzzification (valeur classique : 2)
    #   - error : critère d'erreur pour la convergence
    #   - maxiter : nombre maximum d'itérations
    #   - seed : graine pour la reproductibilité
      
    # Retourne :
    #   - centers : centres des clusters (shape: [n_clusters, L])
    #   - u : matrice d'appartenance (shape: [n_clusters, n_spectra])
    # """

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(spectra)
    centers = kmeans.cluster_centers_
    
    return centers, labels

def remove_outliers(spectra, memberships, threshold=0.6):
    """
    Supprime les outliers en éliminant les spectres dont le degré d'appartenance maximum
    est inférieur au seuil défini.
    
    Args :
        spectra : array de spectres de forme (n_spectra, L).
        memberships : matrice d'appartenance de forme (n_clusters, n_spectra).
        threshold : seuil minimal pour le degré d'appartenance maximum.
      
    Returns :
        spectra_clean : array contenant uniquement les spectres non considérés comme outliers.
        indices_keep : indices des spectres retenus.
    """
    max_membership = np.max(memberships, axis=0)
    indices_keep = np.where(max_membership >= threshold)[0]
    spectra_clean = spectra[indices_keep]
    return spectra_clean, indices_keep

def group_spectra_by_cluster(spectra, labels):
    """
    Regroupe les spectres en fonction du cluster auquel ils appartiennent.
    Pour chaque spectre, on attribue le cluster pour lequel le degré d'appartenance est maximal.
    
    Paramètres :
        spectra : tableau de spectres de forme (n_spectra, L)
        memberships : matrice d'appartenance de forme (n_clusters, n_spectra)
      
    Retourne :
        groups : liste de tableaux, où groups[i] contient les spectres du cluster i 
        (forme : (nb_spectres_dans_cluster, L)).
    """
    # Attribution du cluster via argmax sur la dimension des clusters
    n_clusters = np.max(labels)+1
    groups = []
    for i in range(n_clusters):
        indices = np.where(labels == i)[0]
        group = spectra[indices, :]
        groups.append(group)
        print(group.shape)
    return groups

def apply_augmentation(groups, cvar=0.4):
    L = groups[0].shape[1]
    aug_spectra = np.zeros((len(groups),L))
    for iGroup in range(len(groups)):
        cur_group = groups[iGroup]
        indChoice = np.random.permutation(cur_group.shape[0])[0]
        spectrum = cur_group[indChoice]
        aug_spectrum = augment_spectrum(spectrum, cvar=cvar)
        aug_spectra[iGroup] = aug_spectrum
        
    return aug_spectra.T

def correct_permutations(A_GT, augmented_A):
    """
    Corrects column permutations of augmented_A to best match A_GT.
    
    Parameters:
        A_GT (numpy.ndarray): Ground truth matrix.
        augmented_A (numpy.ndarray): Augmented matrix with permuted columns.
    
    Returns:
        numpy.ndarray: Permutation-corrected version of augmented_A.
    """
    # Normalize columns of A_GT and augmented_A
    A_GT_norm = A_GT / np.linalg.norm(A_GT, axis=0, keepdims=True)
    augmented_A_norm = augmented_A / np.linalg.norm(augmented_A, axis=0, keepdims=True)
    
    # Compute cost matrix based on negative cosine similarity (maximize alignment)
    cost_matrix = -np.dot(A_GT_norm.T, augmented_A_norm)
    
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Reorder the columns of augmented_A based on the optimal assignment
    corrected_A = augmented_A[:, col_ind]
    
    return corrected_A

GT_available = True
mat_spectres = sio.loadmat('/home/ids/edabier/HSU/datasets/Jasper/A_VCA_Jasper.mat')['donnees']
mat_spectres = np.swapaxes(mat_spectres, 1, 2)
spectra = mat_spectres.reshape([4*10000,198])

# 1. Éliminer les doublons
unique_spectra = remove_duplicates(spectra, tol=1e-4)

# Normalisation
norms = np.linalg.norm(unique_spectra, axis=1, keepdims=True)
unique_spectra_norm = unique_spectra / norms

# 2. Regrouper les spectres par k-means
n_clusters = 4 
centers, memberships = group_spectra_kmeans(unique_spectra_norm, n_clusters=n_clusters)
groups = group_spectra_by_cluster(unique_spectra, memberships)
