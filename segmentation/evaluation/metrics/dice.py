import torch
from typing import Any

def compute(label: Any, prediction: Any, threshold: float = 0.5) -> float:
    """
    Computes the Dice coefficient for binary segmentation using PyTorch.
    
    Args:
        label (Any): Ground truth mask. Can be a numpy array or a torch.Tensor.
        prediction (Any): Predicted mask or probability map. Can be a numpy array or a torch.Tensor.
        threshold (float): Threshold to binarize predictions if they are not already binary.
        
    Returns:
        float: Dice coefficient.
    """
    eps = 1e-8

    # Convert inputs to torch tensors if they are not already
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)

    # Convert tensors to float type
    label = label.float()
    
    # If prediction is not a boolean tensor, binarize using the threshold
    if prediction.dtype != torch.bool:
        prediction_bin = (prediction >= threshold).float()
    else:
        prediction_bin = prediction.float()

    intersection = torch.sum(label * prediction_bin)
    dice = (2.0 * intersection) / (torch.sum(label) + torch.sum(prediction_bin) + eps)
    return dice.item()
