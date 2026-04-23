import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

import src.utils.losses as losses
import src.utils.utils as utils

def plot_losses(total_loss, loss_sad, loss_ab, loss_tv, loss_mse):
    fig = plt.figure(figsize=(10, 8))

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(total_loss)
    ax1.set_title('Training losses')

    ax2 = plt.subplot2grid((3, 2), (1, 0))
    ax2.plot(loss_sad, color='orange')
    ax2.set_title('SAD loss')

    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.plot(loss_ab, color='green')
    ax3.set_title('Abund. reg')

    ax4 = plt.subplot2grid((3, 2), (2, 0))
    ax4.plot(loss_tv, color='red')
    ax4.set_title('TV loss')

    ax5 = plt.subplot2grid((3, 2), (2, 1))
    ax5.plot(loss_mse, color='purple')
    ax5.set_title('MSE loss')

    plt.tight_layout()
    plt.show()

def compute_metrics_and_plot(E_hat=None, A_hat=None, A_gt=None, E_gt=None, model_name=None, normalize_E=True, normalize_A=True, return_results=False, plot_A=True, plot_E=True, hypersigma=False, cmap='viridis', save_mat=None):
    """
    Displays the predicted endmembers and abundances
    """
    n_graph = None

    sad = losses.SADLoss()
    mse = nn.MSELoss(reduction="sum")

    bg_colors = ["mediumpurple", "cornflowerblue", "indianred", "goldenrod", "mediumseagreen", "lightpink"]
    colors = ["thistle", "lavender", "mistyrose", "lightyellow", "lightblue", "lavenderblush"]

    if E_gt != None:
        E_gt = E_gt.detach().cpu()
        E_hat = E_hat.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if E_gt.dim() == 3:
            E_gt = E_gt[0]

        E_hat, A_hat, indices = utils.order_endmembers(E_gt, E_hat, A_hat)

    if E_hat != None:

        E_hat = E_hat.detach().cpu()
        if E_hat.dim() == 3:   
            E_hat = E_hat[0]

        c = E_hat.shape[1]
        n_graph = c // 2
        if c % 2 != 0: n_graph = n_graph + 1
            
        if E_gt != None:
            
            if normalize_E:
                E_hat = utils.normalize(E_hat, is_endmember=True)
                E_gt = utils.normalize(E_gt, is_endmember=True)

            total_sad = sad(E_gt, E_hat)

            if plot_E:
                fig, axes = plt.subplots(2, n_graph, figsize=(7,5))
                axes = axes.flatten()
                for i in range(c):
                    ax = axes[i]

                    sad_val = sad(E_gt[:, i], E_hat[:, i])

                    ax.plot(E_gt[:, i], 'r', linewidth=1.0, label='GT')
                    ax.plot(E_hat[:, i], 'k-', linewidth=1.0, label='predict')
                    ax.set_title(f"SAD = {format(sad_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[0,i].item()], color=colors[indices[0,i].item()])

                    if i == 0:
                        ax.legend() 

                for j in range(i + 1, len(axes)):
                    axes[j].axis('off')

                plt.subplots_adjust(hspace=0.5, wspace=0.4)

                if model_name != None:
                    E_title = f"{model_name} Endmember estimation, SAD = {format(total_sad, '.3f')}"
                else:
                    E_title = f"Endmember estimation, SAD = {format(total_sad, '.3f')}"
                    
                plt.suptitle(E_title)

                if save_mat is not None:
                    os.makedirs(save_mat, exist_ok=True)
                    plt.savefig(f"{save_mat}/E_hat.png", bbox_inches='tight', dpi=300)
            
        else:
            if plot_E:
                if model_name != None:
                    E_title = f'{model_name} Endmember estimation'
                else:
                    E_title = 'Endmember estimation'
                for i in range(c):
                    ax = plt.subplot(2, n_graph, i + 1)
                    plt.plot(E_hat[:, i].detach().cpu(), 'k-', linewidth=1.0, label='predict')
                plt.suptitle(E_title)

                if save_mat is not None:
                    os.makedirs(save_mat, exist_ok=True)
                    plt.savefig(f"{save_mat}/E_hat.png", bbox_inches='tight', dpi=300)

    if A_gt is not None:
        
        A_gt = A_gt.detach().cpu()
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]
        
        if A_gt.dim() == 4:
            A_gt = A_gt[0]

        if n_graph is None:
            c = A_hat.shape[0]
            n_graph = c // 2
            if c % 2 != 0: n_graph = n_graph + 1
        
        total_sad_a = sad(A_gt.flatten(1).T, A_hat.flatten(1).T)

        if normalize_A:
            A_hat = utils.normalize(A_hat)
            A_gt = utils.normalize(A_gt)
        
        if hypersigma:
            total_mse = losses.hypersigma_mse(A_gt, A_hat)

        else:
            total_mse = mse(A_gt, A_hat)/(torch.norm(A_gt)**2)

        if plot_A:
            fig, axes = plt.subplots(2, c, figsize=(10, 5))
            for i in range(c):
                pred = axes[0, i].imshow(A_hat[i].detach().cpu(), cmap=cmap)

                if hypersigma:
                    mse_val = losses.hypersigma_mse(A_gt[i].unsqueeze(0), A_hat[i].unsqueeze(0))
                
                else:
                    mse_val = mse(A_gt[i], A_hat[i])/(torch.norm(A_gt[i])**2)

                axes[0, i].set_title(f"NMSE = {format(mse_val, '.2f')}", fontsize=10, pad=10, backgroundcolor=bg_colors[indices[0,i].item()], color=colors[indices[0,i].item()])
                axes[0, i].axis('off')

                gt = axes[1, i].imshow(A_gt[i].detach().cpu(), cmap=cmap)
                axes[1, i].axis('off')

                fig.colorbar(pred, ax=axes[0, i], fraction=0.046, pad=0.04)
                fig.colorbar(gt, ax=axes[1, i], fraction=0.046, pad=0.04)
            fig.text(0.05, 0.7, 'prediction', va='center', ha='center', fontsize=12, rotation='vertical')
            fig.text(0.05, 0.4, 'gt', va='center', ha='center', fontsize=12, rotation='vertical')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.1, wspace=0.1, hspace=0.5)
            fig.tight_layout(rect=[0.05, 0.25, 0.95, 0.9])

            if model_name != None:
                A_title = f"{model_name} abundance estimation, NMSE = {format(total_mse, '.3f')}"
            else:
                A_title = f"Abundance estimation, NMSE = {format(total_mse, '.3f')}"
            plt.suptitle(A_title)

            if save_mat is not None:
                os.makedirs(save_mat, exist_ok=True)
                plt.savefig(f"{save_mat}/A_hat.png", bbox_inches='tight', dpi=300)

    elif plot_A:
        A_hat = A_hat.detach().cpu()
        if A_hat.dim() == 4:
            A_hat = A_hat[0]

        c = A_hat.shape[0]

        fig, axes = plt.subplots(1, c, figsize=(10, 5))
        for i in range(c):
            pred = axes[i].imshow(A_hat[i].detach().cpu(), cmap=cmap)
            axes[i].axis('off')
            fig.colorbar(pred, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Adaptative placement of the title as a function of the number of endmembers
        offset_y = -(0.1/3) * c + 0.9
        plt.suptitle("Abundance estimation", y=offset_y)

        if save_mat is not None:
            os.makedirs(save_mat, exist_ok=True)
            plt.savefig(f"{save_mat}/A_hat.png", bbox_inches='tight', dpi=300)
    
    if return_results:
        return total_sad, total_sad_a, total_mse
    
def plot_hsi(Y, n_channels, rgb=False, title=None):
    """
    Displays n channels of the input HSI
    Must be of shape (batch, B, H, W) or (B, H, W)
    """
    if Y.dim() > 3:
        Y = Y.squeeze(0)

    fig, axes = plt.subplots(1, n_channels, figsize=(20, 20))
    B, H, W = Y.shape
    step = (B - 1) / (n_channels - 1)

    for k, idx in enumerate(range(n_channels)):
        i = int(k*step)

        if rgb:
            im = axes[idx].imshow(Y[i:i+3].permute(1,2,0).detach().cpu())
        else:
            im = axes[idx].imshow(Y[i].detach().cpu())
            fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        axes[idx].set_title(f"Channel {i} / {B}")
        
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
    
    if title != None:
        plt.suptitle(title, y=0.6)
    # plt.savefig(f"/home/edabier/Documents/Thèse/benchmark/DOFA_shift_features/{title}.png")

def compare_hsis(Y_gt, Y_hat, title=None, gt_name=None, hat_name=None, n=4):
    """
    Displays the first 4 channels of both reconstructed and groundtruth HSIs
    Must be of shape (batch, B, H, W) or (B, H, W)
    """
    if Y_gt.dim() > 3:
        Y_gt = Y_gt.squeeze(0)
    if Y_hat.dim() > 3:
        Y_hat = Y_hat.squeeze(0)

    fig, axes = plt.subplots(2, n, figsize=(10, 5))
    B, H, W = Y_gt.shape

    if gt_name is not None:
        axes[1, 0].set_title(f"{gt_name}", fontsize=12)

    else:
        axes[1, 0].set_title("GT", fontsize=12)

    if hat_name is not None:
        axes[0, 0].set_title(f"{hat_name}", fontsize=12)

    else:
        axes[0, 0].set_title("Prediction", fontsize=12)

    for i in range(n):
        i_th = int((i/n)*B)
        pred = axes[0, i].imshow(Y_hat[i_th].detach().cpu())
        axes[0, i].axis('off')

        gt = axes[1, i].imshow(Y_gt[i_th].detach().cpu())
        axes[1, i].axis('off')
    if title != None:
        plt.suptitle(title)
    bar = plt.colorbar(gt)
    bar = plt.colorbar(pred)
