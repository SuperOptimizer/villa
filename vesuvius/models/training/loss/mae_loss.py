import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedReconstructionLoss(nn.Module):
    """
    Masked-region reconstruction loss for 2-D or 3-D MAE style pre-training.

    Args
    ----
    base_loss         :  'mse' | 'l1' | 'smooth_l1'
    normalize_targets :  normalise pred/target per-sample using *visible* voxels
    reduction         :  'mean' (masked-mean), 'sum', or 'none'
    eps               :  small constant for numerical stability
    variance_threshold:  minimum std deviation required for normalization (default: 0.1)
                        If std < threshold, normalization is skipped to avoid numerical instability
    """

    def __init__(self,
                 base_loss: str = 'mse',
                 normalize_targets: bool = True,
                 reduction: str = 'mean',
                 eps: float = 1e-6,
                 variance_threshold: float = 0.1):
        super().__init__()

        # choose base loss
        if base_loss == 'mse':
            self.base_loss_fn = F.mse_loss
        elif base_loss == 'l1':
            self.base_loss_fn = F.l1_loss
        elif base_loss == 'smooth_l1':
            self.base_loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f'Unknown base_loss: {base_loss}')

        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError(f'Unknown reduction: {reduction}')

        self.normalize_targets = normalize_targets
        self.reduction        = reduction
        self.eps              = eps
        self.variance_threshold = variance_threshold

    # --------------------------------------------------------------------- #
    # utility                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _check_mask(mask: torch.Tensor, pred: torch.Tensor):
        """
        Ensures mask has shape (B,1,...) and is {0,1}.  Raises otherwise.
        """
        if mask.dim() != pred.dim():
            raise ValueError(f'Mask must have same #dims as pred; '
                             f'got {mask.shape} vs {pred.shape}')

        if mask.shape[0] != pred.shape[0]:
            raise ValueError('Batch dim of mask/pred differ.')

        # force channel==1
        if mask.shape[1] != 1:
            raise ValueError(f'Mask channel dim must be 1; got {mask.shape}')

        if not mask.dtype.is_floating_point:
            mask = mask.float()

        # clamp to {0,1}
        mask = mask.clamp(0, 1)
        return mask

    # --------------------------------------------------------------------- #
    # forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self,
                pred : torch.Tensor,
                target: torch.Tensor,
                mask : torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (B,C,H,W) or (B,C,D,H,W)
        target : same shape as pred
        mask   : (B,1,H,W) or (B,1,D,H,W) – 1 = masked, 0 = visible
        """
        if pred.shape != target.shape:
            raise ValueError(f'pred and target shapes differ: '
                             f'{pred.shape} vs {target.shape}')

        mask = self._check_mask(mask, pred)  # (B,1, ...)

        # ------------------------------------------------------------------ #
        # normalisation (optional)                                           #
        # ------------------------------------------------------------------ #
        if self.normalize_targets:
            visible = 1.0 - mask                 # (B,1, ...)
            dims    = tuple(range(2, pred.dim()))  # spatial dims  reduce over

            # count visible voxels, clamp to avoid /0
            num_vis = visible.sum(dim=dims, keepdim=True).clamp_min_(1.0)

            # μ, σ over *visible* voxels
            mean = (target * visible).sum(dim=dims, keepdim=True) / num_vis
            var  = ((target - mean).pow(2) * visible).sum(dim=dims, keepdim=True) / num_vis
            std  = (var + self.eps).sqrt()

            # Only normalize if std is above threshold to avoid numerical instability
            # Create a mask for samples with sufficient variance
            normalize_mask = (std > self.variance_threshold).float()
            
            # Apply normalization only where std is sufficient
            # For samples with low std, keep original values
            target_normalized = (target - mean) / std
            pred_normalized = (pred - mean) / std
            
            # Use normalized values where std > threshold, original values otherwise
            target = target * (1 - normalize_mask) + target_normalized * normalize_mask
            pred = pred * (1 - normalize_mask) + pred_normalized * normalize_mask

        # ------------------------------------------------------------------ #
        # pixel/voxel-wise base loss                                         #
        # ------------------------------------------------------------------ #
        loss = self.base_loss_fn(pred, target, reduction='none')  # (B,C,...)

        # apply mask  (broadcast channel dim automatically)
        loss = loss * mask

        if self.reduction == 'none':
            return loss

        if self.reduction == 'sum':
            return loss.sum()

        # 'mean' over *masked* voxels only
        num_masked = mask.sum()
        if num_masked > 0:
            return loss.sum() / num_masked
        # nothing masked – return 0 with grad
        return pred.new_tensor(0.0, requires_grad=True)
