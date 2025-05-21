import torch.nn as nn
from .utils import get_pool_and_conv_props, get_n_blocks_per_stage
from models.model.build.encoder import Encoder
from models.model.build.decoder import Decoder

def get_activation_module(activation_str: str):
    act_str = activation_str.lower()
    if act_str == "none":
        return None
    elif act_str == "sigmoid":
        return nn.Sigmoid()
    elif act_str == "softmax":
        print("Warning: Softmax not applicable for single-channel output. Using sigmoid instead.")
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation type: {activation_str}")

class NetworkFromConfig(nn.Module):
    def __init__(self, mgr):
        super().__init__()
        self.mgr = mgr
        self.targets = mgr.targets
        self.patch_size = mgr.train_patch_size
        self.batch_size = mgr.train_batch_size
        self.in_channels = 1 
        self.autoconfigure = mgr.autoconfigure

        if hasattr(mgr, 'model_config') and mgr.model_config:
            model_config = mgr.model_config
        else:
            print("model_config is empty; using inference_config instead")
            model_config = mgr.inference_config

        self.save_config = True

        # --------------------------------------------------------------------
        # Common nontrainable parameters (ops, activation, etc.)
        # --------------------------------------------------------------------
        self.conv_op = model_config.get("conv_op", "nn.Conv3d")
        self.conv_op_kwargs = model_config.get("conv_op_kwargs", {"bias": False})
        self.pool_op = model_config.get("pool_op", "nn.AvgPool3d")
        self.dropout_op = model_config.get("dropout_op", "nn.Dropout3d")
        self.dropout_op_kwargs = model_config.get("dropout_op_kwargs", {"p": 0.0})
        self.norm_op = model_config.get("norm_op", "nn.InstanceNorm3d")
        self.norm_op_kwargs = model_config.get("norm_op_kwargs", {"affine": False, "eps": 1e-5})
        self.conv_bias = model_config.get("conv_bias", False)
        self.nonlin = model_config.get("nonlin", "nn.LeakyReLU")
        self.nonlin_kwargs = model_config.get("nonlin_kwargs", {"inplace": True})

        # Get op_dims from ConfigManager if available, or determine from patch_size
        self.op_dims = getattr(mgr, 'op_dims', None)
        if self.op_dims is None:
            # Fallback to determining from patch size if not set by ConfigManager
            if len(self.patch_size) == 2:
                self.op_dims = 2
                print(f"Using 2D operations based on patch_size {self.patch_size}")
            elif len(self.patch_size) == 3:
                self.op_dims = 3
                print(f"Using 3D operations based on patch_size {self.patch_size}")
            else:
                raise ValueError(f"Patch size must have either 2 or 3 dimensions! Got {len(self.patch_size)}D: {self.patch_size}")
        else:
            print(f"Using dimensionality ({self.op_dims}D) from ConfigManager")
        
        # Convert string operation types to actual PyTorch classes
        if isinstance(self.conv_op, str):
            if self.op_dims == 2:
                self.conv_op = nn.Conv2d
                print("Using 2D convolutions (nn.Conv2d)")
            else:
                self.conv_op = nn.Conv3d
                print("Using 3D convolutions (nn.Conv3d)")
                
        if isinstance(self.pool_op, str):
            if self.op_dims == 2:
                self.pool_op = nn.AvgPool2d
                print("Using 2D pooling (nn.AvgPool2d)")
            else:
                self.pool_op = nn.AvgPool3d
                print("Using 3D pooling (nn.AvgPool3d)")
                
        if isinstance(self.norm_op, str):
            if self.op_dims == 2:
                self.norm_op = nn.InstanceNorm2d
                print("Using 2D normalization (nn.InstanceNorm2d)")
            else:
                self.norm_op = nn.InstanceNorm3d
                print("Using 3D normalization (nn.InstanceNorm3d)")
                
        if isinstance(self.dropout_op, str):
            if self.op_dims == 2:
                self.dropout_op = nn.Dropout2d
                print("Using 2D dropout (nn.Dropout2d)")
            else:
                self.dropout_op = nn.Dropout3d
                print("Using 3D dropout (nn.Dropout3d)")
        if self.nonlin in ["nn.LeakyReLU", "LeakyReLU"]:
            self.nonlin = nn.LeakyReLU
            self.nonlin_kwargs = {"negative_slope": 1e-2, "inplace": True}
        elif self.nonlin in ["nn.ReLU", "ReLU"]:
            self.nonlin = nn.ReLU
            self.nonlin_kwargs = {"inplace": True}

        # --------------------------------------------------------------------
        # Architecture parameters.
        # --------------------------------------------------------------------
        if self.autoconfigure:
            print("--- Autoconfiguring network from config ---")
            self.basic_encoder_block = "BasicBlockD"
            self.basic_decoder_block = "ConvBlock"
            self.bottleneck_block = "BasicBlockD"
            num_pool_per_axis, pool_op_kernel_sizes, conv_kernel_sizes, final_patch_size, must_div = \
                get_pool_and_conv_props(
                    spacing=mgr.spacing,
                    patch_size=self.patch_size,
                    min_feature_map_size=4,
                    max_numpool=999999
                )
            self.num_stages = len(pool_op_kernel_sizes)
            base_features = 32
            max_features = 320
            features = []
            for i in range(self.num_stages):
                feats = base_features * (2 ** i)
                features.append(min(feats, max_features))
            self.features_per_stage = features
            self.n_blocks_per_stage = get_n_blocks_per_stage(self.num_stages)
            self.n_conv_per_stage_decoder = [1] * (self.num_stages - 1)
            self.strides = pool_op_kernel_sizes
            self.kernel_sizes = conv_kernel_sizes
            self.pool_op_kernel_sizes = pool_op_kernel_sizes
        else:
            print("--- Configuring network from config file ---")
            self.basic_encoder_block = model_config.get("basic_encoder_block", "BasicBlockD")
            self.basic_decoder_block = model_config.get("basic_decoder_block", "ConvBlock")
            self.bottleneck_block = model_config.get("bottleneck_block", "BasicBlockD")
            self.features_per_stage = model_config.get("features_per_stage",
                                                        mgr.inference_config.get("features_per_stage",
                                                                                   [32, 64, 128, 256, 320, 320, 320]))
            self.num_stages = model_config.get("n_stages", 7)
            self.n_blocks_per_stage = model_config.get("n_blocks_per_stage", [1, 3, 4, 6, 6, 6, 6])
            # Set default kernel sizes and pool kernel sizes based on dimensionality
            default_kernel = [[3, 3]] * self.num_stages if self.op_dims == 2 else [[3, 3, 3]] * self.num_stages
            default_pool = [[1, 1]] * self.num_stages if self.op_dims == 2 else [[1, 1, 1]] * self.num_stages
            default_strides = [[1, 1]] * self.num_stages if self.op_dims == 2 else [[1, 1, 1]] * self.num_stages
            
            print(f"Using {'2D' if self.op_dims == 2 else '3D'} kernel defaults: {default_kernel[0]}")
            print(f"Using {'2D' if self.op_dims == 2 else '3D'} pool defaults: {default_pool[0]}")
            
            self.kernel_sizes = model_config.get("kernel_sizes", default_kernel)
            self.pool_op_kernel_sizes = model_config.get("pool_op_kernel_sizes", default_pool)
            self.n_conv_per_stage_decoder = model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1))
            self.strides = model_config.get("strides", self.pool_op_kernel_sizes)
            
            # Validate that kernel sizes and strides match the input dimensionality
            for i in range(len(self.kernel_sizes)):
                if len(self.kernel_sizes[i]) != self.op_dims:
                    print(f"WARNING: Kernel size at stage {i} does not match input dimensionality. Fixing...")
                    if self.op_dims == 2:
                        # Convert 3D kernels to 2D by taking first two dimensions
                        if len(self.kernel_sizes[i]) > 2:
                            print(f"  Converting 3D kernel {self.kernel_sizes[i]} to 2D kernel {self.kernel_sizes[i][:2]}")
                            self.kernel_sizes[i] = self.kernel_sizes[i][:2]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 2D kernel [3,3] for stage {i}")
                            self.kernel_sizes[i] = [3, 3]
                    else:
                        # Convert 2D kernels to 3D by adding a dimension
                        if len(self.kernel_sizes[i]) == 2:
                            print(f"  Converting 2D kernel {self.kernel_sizes[i]} to 3D kernel {self.kernel_sizes[i] + [3]}")
                            self.kernel_sizes[i] = self.kernel_sizes[i] + [3]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 3D kernel [3,3,3] for stage {i}")
                            self.kernel_sizes[i] = [3, 3, 3]
                        
            for i in range(len(self.strides)):
                if len(self.strides[i]) != self.op_dims:
                    print(f"WARNING: Stride at stage {i} does not match input dimensionality. Fixing...")
                    if self.op_dims == 2:
                        # Convert 3D strides to 2D by taking first two dimensions
                        if len(self.strides[i]) > 2:
                            print(f"  Converting 3D stride {self.strides[i]} to 2D stride {self.strides[i][:2]}")
                            self.strides[i] = self.strides[i][:2]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 2D stride [1,1] for stage {i}")
                            self.strides[i] = [1, 1]
                    else:
                        # Convert 2D strides to 3D by adding a dimension
                        if len(self.strides[i]) == 2:
                            print(f"  Converting 2D stride {self.strides[i]} to 3D stride {self.strides[i] + [1]}")
                            self.strides[i] = self.strides[i] + [1]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 3D stride [1,1,1] for stage {i}")
                            self.strides[i] = [1, 1, 1]
                        
            for i in range(len(self.pool_op_kernel_sizes)):
                if len(self.pool_op_kernel_sizes[i]) != self.op_dims:
                    print(f"WARNING: Pool kernel size at stage {i} does not match input dimensionality. Fixing...")
                    if self.op_dims == 2:
                        # Convert 3D pool kernels to 2D by taking first two dimensions
                        if len(self.pool_op_kernel_sizes[i]) > 2:
                            print(f"  Converting 3D pool kernel {self.pool_op_kernel_sizes[i]} to 2D pool kernel {self.pool_op_kernel_sizes[i][:2]}")
                            self.pool_op_kernel_sizes[i] = self.pool_op_kernel_sizes[i][:2]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 2D pool kernel [1,1] for stage {i}")
                            self.pool_op_kernel_sizes[i] = [1, 1]
                    else:
                        # Convert 2D pool kernels to 3D by adding a dimension
                        if len(self.pool_op_kernel_sizes[i]) == 2:
                            print(f"  Converting 2D pool kernel {self.pool_op_kernel_sizes[i]} to 3D pool kernel {self.pool_op_kernel_sizes[i] + [1]}")
                            self.pool_op_kernel_sizes[i] = self.pool_op_kernel_sizes[i] + [1]
                        else:
                            # Handle case of incomplete specification
                            print(f"  Setting default 3D pool kernel [1,1,1] for stage {i}")
                            self.pool_op_kernel_sizes[i] = [1, 1, 1]

        # Derive stem channels from first feature map if not provided.
        self.stem_n_channels = self.features_per_stage[0]

        # --------------------------------------------------------------------
        # Build network.
        # --------------------------------------------------------------------
        self.shared_encoder = Encoder(
            input_channels=self.in_channels,
            basic_block=self.basic_encoder_block,
            n_stages=self.num_stages,
            features_per_stage=self.features_per_stage,
            n_blocks_per_stage=self.n_blocks_per_stage,
            bottleneck_block=self.bottleneck_block,
            conv_op=self.conv_op,
            kernel_sizes=self.kernel_sizes,
            conv_bias=self.conv_bias,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
            strides=self.strides,
            return_skips=True,
            do_stem=model_config.get("do_stem", True),
            stem_channels=model_config.get("stem_channels", self.stem_n_channels),
            bottleneck_channels=model_config.get("bottleneck_channels", None),
            stochastic_depth_p=model_config.get("stochastic_depth_p", 0.0),
            squeeze_excitation=model_config.get("squeeze_excitation", False),
            squeeze_excitation_reduction_ratio=model_config.get("squeeze_excitation_reduction_ratio", 1.0/16.0)
        )
        self.task_decoders = nn.ModuleDict()
        self.task_activations = nn.ModuleDict()
        for target_name, target_info in self.targets.items():
            # Use single channel output for binary segmentation
            target_info["out_channels"] = 1
            out_channels = 1  # Single channel for binary segmentation
            # Default to sigmoid activation for binary segmentation if none specified
            activation_str = target_info.get("activation", "sigmoid")
            self.task_decoders[target_name] = Decoder(
                encoder=self.shared_encoder,
                basic_block=model_config.get("basic_decoder_block", "ConvBlock"),
                num_classes=out_channels,
                n_conv_per_stage=model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
                deep_supervision=False
            )
            self.task_activations[target_name] = get_activation_module(activation_str)

        # --------------------------------------------------------------------
        # Build final configuration snapshot.
        # --------------------------------------------------------------------
        
        self.final_config = {
            "model_name": self.mgr.model_name,
            "basic_encoder_block": self.basic_encoder_block,
            "basic_decoder_block": model_config.get("basic_decoder_block", "ConvBlock"),
            "bottleneck_block": self.bottleneck_block,
            "features_per_stage": self.features_per_stage,
            "num_stages": self.num_stages,
            "n_blocks_per_stage": self.n_blocks_per_stage,
            "n_conv_per_stage_decoder": model_config.get("n_conv_per_stage_decoder", [1] * (self.num_stages - 1)),
            "kernel_sizes": self.kernel_sizes,
            "pool_op": self.pool_op.__name__ if hasattr(self.pool_op, "__name__") else self.pool_op,
            "pool_op_kernel_sizes": self.pool_op_kernel_sizes,
            "conv_op": self.conv_op.__name__ if hasattr(self.conv_op, "__name__") else self.conv_op,
            "conv_bias": self.conv_bias,
            "norm_op": self.norm_op.__name__ if hasattr(self.norm_op, "__name__") else self.norm_op,
            "norm_op_kwargs": self.norm_op_kwargs,
            "dropout_op": self.dropout_op.__name__ if hasattr(self.dropout_op, "__name__") else self.dropout_op,
            "dropout_op_kwargs": self.dropout_op_kwargs,
            "nonlin": self.nonlin.__name__ if hasattr(self.nonlin, "__name__") else self.nonlin,
            "nonlin_kwargs": self.nonlin_kwargs,
            "strides": self.strides,
            "return_skips": model_config.get("return_skips", True),
            "do_stem": model_config.get("do_stem", True),
            "stem_channels": model_config.get("stem_channels", self.stem_n_channels),
            "bottleneck_channels": model_config.get("bottleneck_channels", None),
            "stochastic_depth_p": model_config.get("stochastic_depth_p", 0.0),
            "squeeze_excitation": model_config.get("squeeze_excitation", False),
            "squeeze_excitation_reduction_ratio": model_config.get("squeeze_excitation_reduction_ratio", 1.0/16.0),
            "op_dims": self.op_dims,
            "patch_size": self.patch_size,
            "batch_size": self.batch_size,
            "in_channels": self.in_channels,
            "autoconfigure": self.autoconfigure,
            "targets": self.targets
        }

        print("NetworkFromConfig initialized with final configuration:")
        for k, v in self.final_config.items():
            print(f"  {k}: {v}")

    def forward(self, x):
        skips = self.shared_encoder(x)
        results = {}
        for task_name, decoder in self.task_decoders.items():
            logits = decoder(skips)
            activation_fn = self.task_activations[task_name]
            if activation_fn is not None and not self.training:
                logits = activation_fn(logits)
            results[task_name] = logits
        return results
