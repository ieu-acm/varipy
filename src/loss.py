import torch


def dice_coef(y_true: torch.Tensor,
              y_pred: torch.Tensor,
              smooth: float = 1e-6) -> torch.Tensor:
    """ Dice coefficient PyTorch implementation

    Args:
        y_true (torch.Tensor): Label in Dataset
        y_true (torch.Tensor): Model prediction
        smooth (float): To Prevent ZeroDivisionError and gradient explosion, \
            add smooth variable

    Returns:
        (torch.Tensor): Dice Coefficient Loss
    """
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)

    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)

    dice_coef = (2. * intersection + smooth) / (union + smooth)
    dice_coef_loss = 1. - dice_coef

    return dice_coef_loss
