from typing import Union, Tuple, List, Callable
import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class ColorFunctionExtractor:
    def __init__(self, rectangle_value):
        self.rectangle_value = rectangle_value

    def __call__(self, x):
        if np.isscalar(self.rectangle_value):
            return self.rectangle_value
        elif callable(self.rectangle_value):
            return self.rectangle_value(x)
        elif isinstance(self.rectangle_value, (tuple, list)):
            return np.random.uniform(*self.rectangle_value)
        else:
            raise RuntimeError("unrecognized format for rectangle_value")

class BlankRectangleTransform(BasicTransform):
    """
    Overwrites areas in tensors with rectangles of specified intensity.
    Supports nD data.
    """
    def __init__(self, 
                 rectangle_size: Union[int, Tuple, List],
                 rectangle_value: Union[int, Tuple, List, Callable],
                 num_rectangles: Union[int, Tuple[int, int]],
                 force_square: bool = False,
                 p_per_sample: float = 0.5,
                 p_per_channel: float = 0.5):
        """
        Args:
            rectangle_size: Can be:
                - int: creates squares with edge length rectangle_size
                - tuple/list of int: constant size for rectangles
                - tuple/list of tuple/list: ranges for each dimension
            rectangle_value: Intensity value for rectangles. Can be:
                - int: constant value
                - tuple: range for uniform sampling
                - callable: function to determine value
            num_rectangles: Number of rectangles per image
            force_square: If True, only produces squares
            p_per_sample: Probability per sample
            p_per_channel: Probability per channel
        """
        super().__init__()
        self.rectangle_size = rectangle_size
        self.num_rectangles = num_rectangles
        self.force_square = force_square
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.color_fn = ColorFunctionExtractor(rectangle_value)

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _get_rectangle_size(self, img_shape: Tuple[int, ...]) -> List[int]:
        img_dim = len(img_shape)
        
        if isinstance(self.rectangle_size, int):
            return [self.rectangle_size] * img_dim
        
        elif isinstance(self.rectangle_size, (tuple, list)) and all([isinstance(i, int) for i in self.rectangle_size]):
            return list(self.rectangle_size)
        
        elif isinstance(self.rectangle_size, (tuple, list)) and all([isinstance(i, (tuple, list)) for i in self.rectangle_size]):
            if self.force_square:
                return [np.random.randint(self.rectangle_size[0][0], self.rectangle_size[0][1] + 1)] * img_dim
            else:
                return [np.random.randint(self.rectangle_size[d][0], self.rectangle_size[d][1] + 1) 
                        for d in range(img_dim)]
        else:
            raise RuntimeError("unrecognized format for rectangle_size")

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        result = img.clone()
        img_shape = img.shape[1:]  # DHW
        
        if np.random.uniform() < self.p_per_sample:
            for c in range(img.shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    # Number of rectangles
                    n_rect = (self.num_rectangles if isinstance(self.num_rectangles, int) 
                            else np.random.randint(self.num_rectangles[0], self.num_rectangles[1] + 1))
                    
                    for _ in range(n_rect):
                        rectangle_size = self._get_rectangle_size(img_shape)
                        
                        # Get random starting positions
                        lb = [np.random.randint(0, max(img_shape[i] - rectangle_size[i], 1)) 
                            for i in range(len(img_shape))]
                        ub = [i + j for i, j in zip(lb, rectangle_size)]
                        
                        # Create slice for the rectangle
                        my_slice = tuple([c, *[slice(i, j) for i, j in zip(lb, ub)]])
                        
                        # Get intensity value and convert to torch tensor before assignment
                        intensity = self.color_fn(result[my_slice].cpu().numpy())
                        intensity = torch.tensor(intensity, device=result.device, dtype=result.dtype)
                        result[my_slice] = intensity
        
        return result

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **kwargs) -> torch.Tensor:
        return segmentation  # Don't modify segmentations

    def _apply_to_dist_map(self, dist_map: torch.Tensor, **kwargs) -> torch.Tensor:
        # DO NOT blank anything in the distance map
        # (this is an intensity transform, not geometric)
        return dist_map

    def _apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **kwargs):
        raise NotImplementedError

    def _apply_to_regr_target(self, regr_target: torch.Tensor, **kwargs) -> torch.Tensor:
        return regr_target  # Don't modify regression targets
    
import numpy as np
import torch
from typing import Union, Tuple
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

def augment_rician_noise(data: torch.Tensor, noise_variance: Tuple[float, float]) -> torch.Tensor:
    """
    Adds Rician noise to the input tensor.
    
    Args:
        data: Input tensor
        noise_variance: Range for variance of the Gaussian distributions
        
    Returns:
        Tensor with added Rician noise
    """
    variance = np.random.uniform(*noise_variance)
    
    # Generate two independent Gaussian distributions
    noise1 = torch.normal(0, np.sqrt(variance), size=data.shape)
    noise2 = torch.normal(0, np.sqrt(variance), size=data.shape)
    
    # Calculate Rician noise
    return torch.sqrt((data + noise1) ** 2 + noise2 ** 2)

class RicianNoiseTransform(BasicTransform):
    """
    Adds Rician noise with the given variance.
    The Noise of MRI data tends to have a Rician distribution: 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2254141/
    
    Args:
        noise_variance: Tuple of (min, max) for variance of Gaussian distributions
        p_per_sample: Probability of applying the transform per sample
    """
    def __init__(self, 
                 noise_variance: Union[Tuple[float, float], float] = (0, 0.1),
                 p_per_sample: float = 1.0):
        super().__init__()
        self.noise_variance = noise_variance if isinstance(noise_variance, tuple) else (0, noise_variance)
        self.p_per_sample = p_per_sample

    def get_parameters(self, **data_dict) -> dict:
        return {}

    def _apply_to_image(self, img: torch.Tensor, **kwargs) -> torch.Tensor:
        if np.random.uniform() < self.p_per_sample:
            return augment_rician_noise(img, self.noise_variance)
        return img

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **kwargs) -> torch.Tensor:
        return segmentation  # Don't apply noise to segmentations

    def _apply_to_bbox(self, bbox, **kwargs):
        raise NotImplementedError

    def _apply_to_keypoints(self, keypoints, **kwargs):
        raise NotImplementedError

    def _apply_to_regr_target(self, regr_target: torch.Tensor, **kwargs) -> torch.Tensor:
        return regr_target  # Don't apply noise to regression targets