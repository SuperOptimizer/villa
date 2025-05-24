import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLoss(nn.Module):
    """
    Base class for masked losses that only compute loss on specified regions.
    """
    def __init__(self, base_loss_fn):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        
    def forward(self, input, target, loss_mask=None):
        loss = self.base_loss_fn(input, target)
        
        if loss_mask is not None:
            if loss_mask.ndim > loss.ndim:
                loss_mask = loss_mask.squeeze(1)
            elif loss_mask.ndim < loss.ndim:
                loss_mask = loss_mask.unsqueeze(1)
            
            valid_elements = loss_mask.sum().clamp(min=1e-8)
            return (loss * loss_mask).sum() / valid_elements
        else:
            # No mask provided, compute loss over the entire image
            return loss.mean()

class BCEWithLogitsMaskedLoss(MaskedLoss):
    """
    Wrapper for BCEWithLogitsLoss that supports a loss_mask parameter.
    """
    def __init__(self, **kwargs):
        # Always use reduction='none' to support masking
        kwargs['reduction'] = 'none'
        base_loss = nn.BCEWithLogitsLoss(**kwargs)
        super().__init__(base_loss)

class BCEMaskedLoss(MaskedLoss):
    """
    Wrapper for BCE that supports a loss_mask parameter.
    Expects input to already have sigmoid applied.
    """
    def __init__(self, **kwargs):
        # Always use reduction='none' to support masking
        kwargs['reduction'] = 'none'
        base_loss = nn.BCELoss(**kwargs)
        super().__init__(base_loss)

class CrossEntropyMaskedLoss(nn.Module):
    """
    Wrapper for CrossEntropyLoss that supports a loss_mask parameter.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # Always use reduction='none' to support masking
        kwargs['reduction'] = 'none'
        self.ce_loss = nn.CrossEntropyLoss(**kwargs)
        
    def forward(self, input, target, loss_mask=None):
        # If target has shape [B,1,H,W], reshape to [B,H,W]
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
            
        target = target.long()  # Ensure target is long tensor for CE
        
        # Apply standard CrossEntropyLoss (with reduction='none')
        loss = self.ce_loss(input, target)
        
        # Apply mask if provided
        if loss_mask is not None:
            # Make sure mask has same shape as loss
            if loss_mask.ndim > loss.ndim:
                loss_mask = loss_mask.squeeze(1)
            
            # Apply mask and take mean of valid elements
            valid_elements = loss_mask.sum().clamp(min=1e-8)
            return (loss * loss_mask).sum() / valid_elements
        else:
            # No mask, just take mean of all elements
            return loss.mean()

class MSEMaskedLoss(MaskedLoss):
    """
    Wrapper for MSE that supports a loss_mask parameter.
    """
    def __init__(self, **kwargs):
        # Always use reduction='none' to support masking
        kwargs['reduction'] = 'none'
        base_loss = nn.MSELoss(**kwargs)
        super().__init__(base_loss)

class L1MaskedLoss(MaskedLoss):
    """
    Wrapper for L1Loss that supports a loss_mask parameter.
    """
    def __init__(self, **kwargs):
        # Always use reduction='none' to support masking
        kwargs['reduction'] = 'none'
        base_loss = nn.L1Loss(**kwargs)
        super().__init__(base_loss)

# Simple Dice Loss for binary segmentation
class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Works with single-channel sigmoid output.
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target, loss_mask=None):
        # Ensure input has been passed through sigmoid if it hasn't been already
        if not torch.all((input >= 0) & (input <= 1)):
            input = torch.sigmoid(input)
        
        input_flat = input.view(input.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        if loss_mask is not None:
            mask_flat = loss_mask.view(loss_mask.size(0), -1)
            input_flat = input_flat * mask_flat
            target_flat = target_flat * mask_flat
        
        intersection = (input_flat * target_flat).sum(dim=1)
        input_sum = input_flat.sum(dim=1)
        target_sum = target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (input_sum + target_sum + self.smooth)
        
        return 1.0 - dice.mean()


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Dice and BCE with Logits Loss for binary segmentation.
    
    This loss function computes a weighted combination of:
    1. Soft Dice Loss - measures overlap between predicted and target regions
    2. BCE with Logits Loss - measures pixel-wise classification accuracy
    
    Args:
        dice_weight (float): Weight for dice loss component. Default: 0.5
        bce_weight (float): Weight for BCE loss component. Default: 0.5
        label_smoothing (float): Label smoothing factor. 0.0 = no smoothing,
                               0.1 = smooth labels to [0.1, 0.9]. Default: 0.0
        smooth (float): Small epsilon for dice loss numerical stability. Default: 1e-5
        
    Example:
        # Equal weighting
        loss_fn = CombinedDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        
        # Favor dice loss
        loss_fn = CombinedDiceBCELoss(dice_weight=0.7, bce_weight=0.3)
        
        # With label smoothing
        loss_fn = CombinedDiceBCELoss(dice_weight=0.6, bce_weight=0.4, label_smoothing=0.1)
    """
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, label_smoothing=0.0, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.label_smoothing = label_smoothing
        
        # Initialize component losses
        self.dice_loss = SoftDiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def apply_label_smoothing(self, target):
        """
        Apply label smoothing to binary targets.
        Maps 0 -> label_smoothing, 1 -> 1 - label_smoothing
        """
        if self.label_smoothing > 0.0:
            # Smooth the labels: 0 becomes label_smoothing, 1 becomes 1-label_smoothing
            smoothed_target = target * (1.0 - 2 * self.label_smoothing) + self.label_smoothing
            return smoothed_target
        return target
        
    def forward(self, input, target, loss_mask=None):
        """
        Forward pass for combined loss.
        
        Args:
            input (torch.Tensor): Raw logits from model (before sigmoid)
            target (torch.Tensor): Ground truth binary labels [0, 1]
            loss_mask (torch.Tensor, optional): Mask to specify valid regions for loss computation
            
        Returns:
            torch.Tensor: Combined weighted loss value
        """
        # Apply label smoothing if specified
        smoothed_target = self.apply_label_smoothing(target)
        
        # Compute dice loss (dice loss expects sigmoid probabilities)
        dice_loss_value = self.dice_loss(input, smoothed_target, loss_mask)
        
        # Compute BCE loss (BCE with logits expects raw logits)
        bce_loss_raw = self.bce_loss(input, smoothed_target)
        
        # Apply mask to BCE loss if provided
        if loss_mask is not None:
            # Ensure mask has same shape as BCE loss
            if loss_mask.ndim > bce_loss_raw.ndim:
                loss_mask = loss_mask.squeeze(1)
            elif loss_mask.ndim < bce_loss_raw.ndim:
                loss_mask = loss_mask.unsqueeze(1)
                
            # Apply mask and compute mean over valid elements
            valid_elements = loss_mask.sum().clamp(min=1e-8)
            bce_loss_value = (bce_loss_raw * loss_mask).sum() / valid_elements
        else:
            # No mask, take mean over all elements
            bce_loss_value = bce_loss_raw.mean()
        
        # Combine losses with specified weights
        combined_loss = (self.dice_weight * dice_loss_value + 
                        self.bce_weight * bce_loss_value)
        
        return combined_loss
