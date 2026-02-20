import torch.nn as nn
import torch
import torch.nn.functional as F

class TVLoss(nn.Module):
    def __init__(self, reduction=None):
        super(TVLoss,self).__init__()
        self.reduction = reduction

    def forward(self,x):
        """
        Expects input x to be of shape (batch, B, H, W)
        """
        batch = x.shape[0]

        diff1 = x[..., 1:, :] - x[..., :-1, :]
        diff2 = x[..., :, 1:] - x[..., :, :-1]

        res1 = diff1.abs().sum([1, 2, 3])
        res2 = diff2.abs().sum([1, 2, 3])
        score = res1 + res2

        if self.reduction == "mean":
            return score.sum() / batch
        elif self.reduction == "sum":
            return score.sum()
        elif self.reduction is None or batch == "none":
            return score[0]

class SADLoss(nn.Module):
    """
    SAD loss function for EndMember matrices. To use it on Abundances, transpose the two inputs. (Doesn't correct permutations)
    """
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if y_pred.dim() == 1:
            y_true = F.normalize(y_true, dim=0, p=2)
            y_pred = F.normalize(y_pred, dim=0, p=2)
        else:
            y_true = F.normalize(y_true, dim=1, p=2)
            y_pred = F.normalize(y_pred, dim=1, p=2)

        A = torch.mul(y_true, y_pred)

        if y_true.dim() == 1:
            A = torch.sum(A, dim=0)
        else:
            A = torch.sum(A, dim=1)
            
        sad = torch.acos(A)
        loss = torch.mean(sad)
        return loss

def hypersigma_mse(Y_gt, Y_hat):
    # y_true:[num_em,H,W]
    # y_pred:[num_em,H,W]

    c = Y_gt.shape[0]
    Y_gt = torch.reshape(Y_gt , [c, -1])
    Y_hat = torch.reshape(Y_hat , [c, -1])

    R = Y_hat - Y_gt
    r = R*R
    mse = torch.mean(r, axis=1)
    mse = torch.sum(mse) / len(mse)
    return mse

