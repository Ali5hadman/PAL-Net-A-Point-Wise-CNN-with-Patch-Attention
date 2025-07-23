
import torch
import torch.nn as nn

def pairwise_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """
    tensor1, tensor2: (batch_size, num_landmarks, dims)
    returns: (batch_size, num_landmarks, num_landmarks) of Euclidean distances
    """
    # [B, N, 1, D] - [B, 1, N, D] → [B, N, N, D]
    diff = tensor1.unsqueeze(2) - tensor2.unsqueeze(1)
    # norm over last dim → [B, N, N]
    return torch.norm(diff, dim=-1)

def distance_error_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """
    Loss = mean absolute difference between all pairwise distances
    """
    gt_d = pairwise_distance(gt, gt)
    pred_d = pairwise_distance(pred, pred)
    return torch.abs(gt_d - pred_d).mean()

def localization_error_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Loss = mean Euclidean distance per landmark
    """
    # shape [B, N]
    errors = torch.norm(y_pred - y_true, dim=-1)
    return errors.mean()

def combined_loss(y_true: torch.Tensor, y_pred: torch.Tensor,
                  alpha: float = 0.5, beta: float = 0.5) -> torch.Tensor:
    loc = localization_error_loss(y_true, y_pred)
    dist = distance_error_loss(y_true, y_pred)
    return alpha * loc + beta * dist

# (Optional) wrap as an nn.Module
class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return combined_loss(y_true, y_pred, self.alpha, self.beta)

class localizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return localization_error_loss(y_true, y_pred)
