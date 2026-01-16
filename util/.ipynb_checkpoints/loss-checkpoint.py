# util/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss (GCE)
    Equation (4-9) in the document.
    """
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, pred, labels):
        # pred: logits
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, num_classes=pred.shape[1]).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class MixupMSELoss(nn.Module):
    """
    Mean Square Error Loss for Mixup (Equation 4-8)
    Constraints the consistency between the prediction and the mixed label distribution.
    """
    def __init__(self):
        super(MixupMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y_a, y_b, lam):
        # pred: logits -> softmax -> probability distribution
        pred_probs = F.softmax(pred, dim=1)
        
        # Create mixed target distribution
        # y_a, y_b are indices, need to convert to one-hot
        num_classes = pred.size(1)
        y_a_one_hot = F.one_hot(y_a, num_classes).float()
        y_b_one_hot = F.one_hot(y_b, num_classes).float()
        
        target_probs = lam * y_a_one_hot + (1 - lam) * y_b_one_hot
        
        # Calculate MSE
        return self.mse(pred_probs, target_probs)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)