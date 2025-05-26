import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU activation function: SwiGLU(x) = Swish(x1) * x2
    where x1, x2 are the first and second half of the input channels
    
    Reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Split input in half along the channel dimension
        x1, x2 = x.chunk(2, dim=self.dim)
        # Apply Swish (SiLU) to first half, multiply by second half
        return F.silu(x1) * x2


class SwiGLUBlock(nn.Module):
    """
    A block that doubles channels, applies SwiGLU, then projects back
    to maintain same output channels as input.
    
    This allows SwiGLU to be used as a drop-in replacement for other
    activation functions without changing the network architecture.
    """
    def __init__(self, channels, conv_op, bias=False):
        super().__init__()
        # Double the channels for SwiGLU
        self.expand = conv_op(channels, channels * 2, kernel_size=1, bias=bias)
        self.swiglu = SwiGLU(dim=1)  # Channel dimension is typically 1
        
    def forward(self, x):
        return self.swiglu(self.expand(x))


class GLU(nn.Module):
    """
    Gated Linear Unit: GLU(x) = x1 * sigmoid(x2)
    where x1, x2 are the first and second half of the input channels
    
    Reference: https://arxiv.org/abs/1612.08083
    """
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Split input in half along the channel dimension
        x1, x2 = x.chunk(2, dim=self.dim)
        # Apply sigmoid to second half, multiply by first half
        return x1 * torch.sigmoid(x2)


class GLUBlock(nn.Module):
    """
    A block that doubles channels, applies GLU, then maintains output channels.
    Similar to SwiGLUBlock but with GLU activation.
    """
    def __init__(self, channels, conv_op, bias=False):
        super().__init__()
        # Double the channels for GLU
        self.expand = conv_op(channels, channels * 2, kernel_size=1, bias=bias)
        self.glu = GLU(dim=1)  # Channel dimension is typically 1
        
    def forward(self, x):
        return self.glu(self.expand(x))
