import os
import yaml
import tensorstore as ts
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
import requests
import zarr
import nrrd
import tempfile
from PIL import Image
from io import BytesIO
from pathlib import Path
from setup.accept_terms import get_installation_path
from data.io.paths import list_files, is_aws_ec2_instance
import torch

# Remove the PIL image size limit
Image.MAX_IMAGE_PIXELS = None

# Function to get the maximum value of a dtype
def get_max_value(dtype: np.dtype) -> Union[float, int]:
    """
    Get the maximum value for a given NumPy dtype.

    Parameters:
    ----------
    dtype : np.dtype
        The NumPy data type to evaluate.

    Returns:
    -------
    Union[float, int]
        The maximum value that the dtype can hold.

    Raises:
    ------
    ValueError
        If the dtype is not a floating point or integer.
    """

    if np.issubdtype(dtype, np.floating):
        max_value = np.finfo(dtype).max
    elif np.issubdtype(dtype, np.integer):
        max_value = np.iinfo(dtype).max
    else:
        raise ValueError("Unsupported dtype")
    return max_value
    
class Volume:
    """
    A class to represent a 3D volume in a scroll or segment.

    Attributes
    ----------
    type : Union[str, int]
        The type of volume, either a scroll or a segment.
    scroll_id : Optional[int]
        ID of the scroll.
    energy : Optional[int]
        Energy value associated with the volume.
    resolution : Optional[float]
        Resolution of the volume.
    segment_id : Optional[int]
        ID of the segment.
    cache : bool
        Indicates if caching is enabled.
    cache_pool : int
        Size of the cache pool.
    normalize : bool
        Indicates if the data should be normalized.
    verbose : bool
        If True, prints additional information during initialization.
    domain : str
        The domain from where data is fetched: 'dl.ash2txt' or 'local'.
    path : Optional[str]
        Path to the local data if domain is 'local'.
    configs : str
        Path to the configuration file.
    url : str
        URL to access the volume data.
    metadata : Dict[str, Any]
        Metadata related to the volume.
    data : List[ts.TensorStore]
        Loaded volume data.
    inklabel : np.ndarray
        Ink label data (only for segments).
    dtype : np.dtype
        Data type of the volume.
    """
        
    def __init__(self, type: Union[str,int],
                 scroll_id: Optional[Union[int, str]] = None,
                 energy: Optional[int] = None,
                 resolution: Optional[float] = None,
                 segment_id: Optional[int] = None,
                 cache: bool = True, cache_pool: int = 1e10,
                 normalize: bool = False,
                 normalization_scheme: str = 'none',
                 return_as_type: str = 'none', # none in this parameter indicates no dtype conversion will occur
                 return_as_tensor: bool = False,
                 verbose : bool = False,
                 domain: Optional[str] = None,
                 path: Optional[str] = None,
                 use_fsspec: bool = False) -> None:
        """
        Initialize the Volume object.

        Parameters
        ----------
        type : Union[str, int]
            The type of volume, either a scroll or a segment. One can also feed directly the canonical scroll, e.g. "Scroll1" or the segment timestamp.
        scroll_id : Optional[Union[int, str]], default = None
            ID of the scroll.
        energy : Optional[int], default = None
            Energy value associated with the volume.
        resolution : Optional[float], default = None
            Resolution of the volume.
        segment_id : Optional[int], default = None
            ID of the segment.
        cache : bool, default = True
            Indicates if caching is enabled.
        cache_pool : int, default = 1e10
            Size of the cache pool in bytes.
        normalize : bool, default = False
            Indicates if the data should be normalized to float  (values between 0 and 1)
        normalization_scheme : str, default = 'none'
            In addition to normalizing to float values, data can be additionally normalized by performing one of the normalization schemes,
            available options are basic standard deviation , and zscore
        return_as_type : str, default = 'none'
            Specify the type of data you'd like returned, options are uint8, uint16, float16, float32. 'none' will simply return in the dtype provided
        return_as_tensor: bool, default = False
            If True, returns the data as a PyTorch tensor instead of a NumPy array
        verbose : bool, default = False
            If True, prints additional information during initialization.
        domain : str, default = "dl.ash2txt"
            The domain from where data is fetched: 'dl.ash2txt' or 'local'.
        path : Optional[str], default = None
            Path to the local data if domain is 'local'.
        use_fsspec : bool, default = False
            If True, uses fsspec instead of TensorStore for data access, which may be faster in some cases.

        Raises
        ------
        ValueError
            If the provided `type` or `domain` is invalid.
        """

        try:
            type = str(type).lower()
            # Store path for later use in get_url_from_yaml
            self.path = path
            
            # Check if this is a zarr file/path
            if type == "zarr":
                self.type = "zarr"
                self.scroll_id = scroll_id
                self.segment_id = segment_id
                # path will be used directly for data access
            
            # Check if this is a segment ID (numeric)
            elif type[0].isdigit():
                scroll_id, energy, resolution, _ = self.find_segment_details(str(type))
                segment_id = int(type)
                type = "segment"
                self.type = type
                self.segment_id = segment_id
                self.scroll_id = scroll_id
            
            # Check if this is a scroll identifier
            elif type.startswith("scroll") and (len(type) > 6):
                self.type = "scroll"
                if (type[6:].isdigit()):
                    self.scroll_id = int(type[6:])
                else:
                    self.scroll_id = str(type[6:])
                self.segment_id = None
            
            else:
                # Handle standard types
                assert type in ["scroll", "segment"], "type should be either 'scroll', 'scroll#', 'segment', or 'zarr'"
                self.type = type

                if type == "segment":
                    assert isinstance(segment_id, int), "segment_id must be an int when type is 'segment'"
                    self.segment_id = segment_id
                    self.scroll_id = scroll_id
                else:
                    self.segment_id = None
                    self.scroll_id = scroll_id

            if domain is None:
                if is_aws_ec2_instance():
                    self.aws = True
                    domain = "local"
                else:
                    self.aws = False
                    domain = "dl.ash2txt"
            else:
                self.aws = False

            assert domain in ["dl.ash2txt", "local"], "domain should be dl.ash2txt or local"

            install_path = get_installation_path()
            
            # Try different possible locations for the config file
            possible_paths = [
                os.path.join(install_path, 'setup', 'configs', f'scrolls.yaml'),  # For editable installs
                os.path.join(install_path, 'vesuvius', 'setup', 'configs', f'scrolls.yaml'),  # For regular installs
                os.path.join(install_path, 'configs', f'scrolls.yaml')  # Fallback
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.configs = path
                    break
            else:
                # If no path exists, use the first one and we'll handle the error later
                self.configs = possible_paths[0]
                if self.verbose:
                    print(f"Warning: Could not find config file at any of the expected locations: {possible_paths}")

            if energy:
                self.energy = energy
            else:
                self.energy = self.grab_canonical_energy()

            if resolution:
                self.resolution = resolution
            else:
                self.resolution = self.grab_canonical_resolution()

            self.domain = domain
            self.cache = cache
            self.cache_pool = cache_pool
            self.normalize = normalize
            self.normalization_scheme = normalization_scheme
            self.return_as_type = return_as_type
            self.return_as_tensor = return_as_tensor
            self.verbose = verbose
            self.use_fsspec = use_fsspec
            
            if self.domain == "dl.ash2txt":
                # For remote paths, get the URL from the config
                self.url = self.get_url_from_yaml()
                if self.verbose:
                    print(f"Using URL from config: {self.url}")
                
                # Load remote data and metadata using either fsspec or TensorStore
                self.metadata = self.load_ome_metadata()
                self.data = self.load_data()
                
                # Handle dtype depending on whether we're using fsspec or TensorStore
                if self.use_fsspec:
                    if self.normalize:
                        self.max_dtype = get_max_value(self.data[0].dtype)
                    self.dtype = self.data[0].dtype
                else:
                    if self.normalize:
                        self.max_dtype = get_max_value(self.data[0].dtype.numpy_dtype)
                    self.dtype = self.data[0].dtype.numpy_dtype
                    
            elif self.domain == "local":
                if self.aws is False:
                    # When not on AWS EC2, a path must be provided for local domain
                    assert path is not None, "For local domain, path must be provided unless running on AWS EC2"
                    self.url = path
                if path is None:
                    # On AWS, get the local path from yaml config
                    self.url = self.get_url_from_yaml()
                
                if self.verbose:
                    print(f"Opening local zarr store at: {self.url}")
                
                # Open the zarr store - lazily, don't load entire file into memory
                if self.use_fsspec:
                    # For fsspec mode, lazily open with zarr
                    self.data = zarr.open(self.url, mode="r")
                else:
                    # For TensorStore mode, use TensorStore for lazy loading
                    # Create file-based kvstore spec for local files
                    context_spec = {
                        'cache_pool': {
                            "total_bytes_limit": self.cache_pool
                        }
                    } if self.cache else {}
                    
                    kvstore_spec = {
                        'driver': 'file',
                        'path': self.url,
                    }
                    
                    # Common spec for local access
                    spec = {
                        'driver': 'zarr',
                        'kvstore': kvstore_spec,
                        'context': context_spec
                    }
                    
                    if self.verbose:
                        print(f"TensorStore spec for local access: {spec}")
                    
                    # Open using TensorStore for lazy loading
                    try:
                        # First, open using zarr to understand the structure
                        zarr_root = zarr.open(self.url, mode='r')
                        # Collect data arrays in a list
                        sub_volumes = []
                        
                        # Look for groups or arrays
                        for key in sorted(zarr_root.keys()):
                            try:
                                # Try to access this as a data array using TensorStore
                                curr_spec = spec.copy()
                                curr_spec['path'] = key  # Add path within zarr store
                                data = ts.open(curr_spec).result()
                                sub_volumes.append(data)
                                if self.verbose:
                                    print(f"Loaded {key} with shape {data.shape}")
                            except Exception as e:
                                if self.verbose:
                                    print(f"Error loading {key} with TensorStore: {e}")
                        
                        if not sub_volumes:
                            # If no subgroups found, try loading the root itself
                            data = ts.open(spec).result()
                            sub_volumes.append(data)
                            
                        self.data = sub_volumes
                    except Exception as e:
                        if self.verbose:
                            print(f"Error using TensorStore for local zarr: {e}")
                            print("Falling back to direct zarr access")
                        # Fall back to direct zarr access if TensorStore fails
                        self.data = zarr.open(self.url, mode="r")
                        # Ensure it's a list of arrays for consistent access
                        if isinstance(self.data, zarr.Group):
                            self.data = [self.data[k] for k in sorted(self.data.keys())]
                
                self.metadata = self.load_ome_metadata()
                
                # Set dtype properties based on access method
                if hasattr(self.data, 'dtype'):
                    # Single array case
                    if self.normalize:
                        self.max_dtype = get_max_value(self.data.dtype)
                    self.dtype = self.data.dtype
                elif isinstance(self.data, list) and len(self.data) > 0:
                    # List of arrays/volumes case
                    if hasattr(self.data[0], 'dtype'):
                        # Direct dtype access (zarr array)
                        if self.normalize:
                            self.max_dtype = get_max_value(self.data[0].dtype)
                        self.dtype = self.data[0].dtype
                    elif hasattr(self.data[0], 'dtype') and hasattr(self.data[0].dtype, 'numpy_dtype'):
                        # TensorStore dtype access
                        if self.normalize:
                            self.max_dtype = get_max_value(self.data[0].dtype.numpy_dtype)
                        self.dtype = self.data[0].dtype.numpy_dtype
        
            if self.type == "segment":
                self.inklabel = np.zeros(self.shape(0), dtype=np.uint8)
                self.download_inklabel()

            if self.verbose:
                self.meta()
        
        except Exception as e:
            print(f"An error occurred while initializing the Volume class: {e}", end="\n")
            print('Load the canonical scroll 1 with Volume(type="scroll", scroll_id=1, energy=54, resolution=7.91)', end="\n")
            print('If loading another part of the same physical scroll use for instance Volume(type="scroll", scroll_id="1b", energy=54, resolution=7.91)', end="\n")
            print('Load a segment (e.g. 20230827161847) with Volume(type="segment", scroll_id=1, energy=54, resolution=7.91, segment_id=20230827161847)')
            raise
    
    def meta(self) -> None:
        """
        Print metadata information about the volume.

        This method provides information about the resolution and shape of the data at the original and scaled resolutions.
        """

        # Assuming the first dataset is the original resolution
        original_dataset = self.metadata['zattrs']['multiscales'][0]['datasets'][0]
        original_scale = original_dataset['coordinateTransformations'][0]['scale'][0]
        original_resolution = float(self.resolution) * float(original_scale)
        idx = 0
        print(f"Data with original resolution: {original_resolution} um, subvolume idx: {idx}, shape: {self.shape(idx)}")

        # Loop through the datasets to print the scaled resolutions, excluding the first one
        for dataset in self.metadata['zattrs']['multiscales'][0]['datasets'][1:]:
            idx += 1
            scale_factors = dataset['coordinateTransformations'][0]['scale']
            scaled_resolution = float(self.resolution) * float(scale_factors[0])
            print(f"Contains also data with scaled resolution: {scaled_resolution} um, subvolume idx: {idx}, shape: {self.shape(idx)}")

    def find_segment_details(self, segment_id: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]:
        """
        Find the details of a segment given its ID.

        Parameters
        ----------
        segment_id : str
            The ID of the segment to search for.

        Returns
        -------
        Tuple[Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]
            A tuple containing scroll_id, energy, resolution, and segment metadata.

        Raises
        ------
        ValueError
            If the segment details cannot be found.
        """
                
        dictionary = list_files()
        stack = [(list(dictionary.items()), [])]

        while stack:
            items, path = stack.pop()
            
            for key, value in items:
                if isinstance(value, dict):
                    # Check if 'segments' key is present in the current level of the dictionary
                    if 'segments' in value:
                        # Check if the segment_id is in the segments dictionary
                        if segment_id in value['segments']:
                            scroll_id, energy, resolution = path[0], path[1], key
                            return scroll_id, energy, resolution, value['segments'][segment_id]
                    # Add nested dictionary to the stack for further traversal
                    stack.append((list(value.items()), path + [key]))

        return None, None, None, None

    def get_url_from_yaml(self) -> str:
        """
        Retrieve the URL for the volume data from the YAML configuration file.

        Returns
        -------
        str
            The URL for the volume data.

        Raises
        ------
        ValueError
            If the URL cannot be found in the configuration.
        FileNotFoundError
            If the configuration file doesn't exist.
        """
        # For zarr type, use the path directly
        if self.type == 'zarr' and hasattr(self, 'path') and self.path is not None:
            if self.verbose:
                print(f"Using path directly for zarr type: {self.path}")
            return self.path
            
        try:
            # Load the YAML file for scroll/segment types
            with open(self.configs, 'r') as file:
                data: Dict[str, Any] = yaml.safe_load(file)
                
            # Handle empty file or invalid YAML
            if not data:
                error_msg = f"Config file at {self.configs} is empty or invalid. "
                if self.type == 'scroll':
                    error_msg += f"You need to populate it with data for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}"
                else:
                    error_msg += f"You need to populate it with data for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, segment: {self.segment_id}"
                raise ValueError(error_msg)
                
            # Retrieve the URL for the given id, energy, and resolution
            if self.type == 'scroll':
                url: str = data.get(str(self.scroll_id), {}).get(str(self.energy), {}).get(str(self.resolution), {}).get("volume")
            elif self.type == 'segment':
                url: str = data.get(str(self.scroll_id), {}).get(str(self.energy), {}).get(str(self.resolution), {}).get("segments", {}).get(str(self.segment_id))
            else:
                raise ValueError(f"Cannot retrieve URL from config for type: {self.type}")

            if url is None:
                if self.type == 'scroll':
                    raise ValueError(f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}. Make sure these values are in your config file.")
                elif self.type == 'segment':
                    raise ValueError(f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, segment: {self.segment_id}. Make sure these values are in your config file.")
                else:
                    raise ValueError("URL not found in the configuration file.")
                
            return url
            
        except FileNotFoundError:
            error_msg = f"Configuration file not found at {self.configs}. "
            error_msg += "Please make sure the vesuvius package is properly installed with configuration files.\n"
            error_msg += "You can download example config files from: https://github.com/ScrollPrize/villa/tree/main/setup/configs"
            raise FileNotFoundError(error_msg)
    
    def load_ome_metadata(self) -> Dict[str, Any]:
        """
        Load the OME (Open Microscopy Environment) metadata for the volume.

        Returns
        -------
        Dict[str, Any]
            The loaded metadata.

        Raises
        ------
        requests.RequestException
            If there is an error loading the metadata from the server.
        """
        try:
            if self.domain == "dl.ash2txt":
                # Load the .zattrs metadata
                # Fix URL format - remove trailing slash if present
                base_url = self.url.rstrip("/")
                zattrs_url = f"{base_url}/.zattrs"
                
                if self.verbose:
                    print(f"Attempting to load metadata from: {zattrs_url}")
                    
                zattrs_response = requests.get(zattrs_url)
                zattrs_response.raise_for_status()
                zattrs = zattrs_response.json()

            elif self.domain == "local":
                # For local domain, access attributes from the zarr store
                if not hasattr(self, 'data') or self.data is None:
                    # This shouldn't happen normally, but just in case
                    if self.verbose:
                        print(f"Warning: Loading data inside load_ome_metadata for local domain")
                    self.data = zarr.open(self.url, mode="r")
                
                zattrs = dict(self.data.attrs)
                
            return {
                "zattrs": zattrs,
            }
        except requests.RequestException as e:
            print(f"Error loading metadata: {e}")
            raise

    def load_data(self):
        """
        Load the data for the volume.

        Returns
        -------
        List[ts.TensorStore] or List[zarr.Array]
            A list of data objects representing the sub-volumes.

        Raises
        ------
        Exception
            If there is an error loading the data from the server.
        """
        # Fix URL format - remove trailing slash if present
        base_url = self.url.rstrip("/")
        sub_volumes = []
        
        # If data is already loaded through direct zarr access, return it
        if hasattr(self, 'data') and self.data is not None and self.domain == "local" and isinstance(self.data, zarr.Group):
            if self.verbose:
                print(f"Data already loaded from local zarr store, using existing data")
            if self.use_fsspec:
                # If it's a zarr group, return a list of its arrays
                return [self.data[k] for k in sorted(self.data.keys())]
            else:
                # Need to convert to TensorStore for each subvolume
                # But this shouldn't happen - we're keeping the same approach throughout
                print("Warning: Mixing data access methods - this should be avoided")
                return self.data
        
        # Determine the access method based solely on use_fsspec flag
        if self.use_fsspec:
            # Use fsspec/zarr for loading (both local and remote)
            import fsspec
            import zarr
            
            # For each resolution level
            for dataset in self.metadata['zattrs']['multiscales'][0]['datasets']:
                path = dataset['path']
                sub_url = f"{base_url}/{path}"
                
                if self.verbose:
                    print(f"Attempting to load data from: {sub_url} using fsspec/zarr")
                
                try:
                    if self.domain == "local":
                        # For local files, open directly with zarr
                        if os.path.exists(sub_url):
                            zarr_array = zarr.open(sub_url, mode='r')
                        else:
                            # Try using the path directly if it exists
                            if os.path.exists(base_url):
                                # This might be a multi-resolution zarr store with the path directly in it
                                try:
                                    zarr_root = zarr.open(base_url, mode='r')
                                    if path in zarr_root:
                                        zarr_array = zarr_root[path]
                                    else:
                                        # Just use the first group or array we find
                                        key = list(zarr_root.keys())[0]
                                        zarr_array = zarr_root[key]
                                except Exception as e:
                                    print(f"Error accessing zarr store at {base_url}: {e}")
                                    raise
                            else:
                                raise FileNotFoundError(f"Cannot find zarr data at {sub_url} or {base_url}")
                    else:
                        # Remote HTTP access through fsspec
                        if self.verbose:
                            print(f"Using fsspec to access remote zarr at {sub_url}")
                        
                        if not sub_url.startswith(('http://', 'https://')):
                            raise ValueError(f"Invalid URL for fsspec HTTP access: {sub_url}")
                            
                        # Create HTTP filesystem with improved performance settings
                        fs = fsspec.filesystem(
                            "http",
                            block_size=2**20,  # 1MB blocks for better throughput
                            cache_type='readahead',  # Prefetch data
                            cache_options={'max_blocks': 32},  # Cache up to 32MB
                        )
                        
                        # Open zarr array via fsspec mapping
                        zarr_map = fsspec.mapping.FSMap(sub_url, fs)
                        zarr_array = zarr.open(zarr_map, mode='r')
                    
                    sub_volumes.append(zarr_array)
                    
                    if self.verbose:
                        print(f"Successfully loaded data from {sub_url} using fsspec/zarr")
                        print(f"Shape: {zarr_array.shape}, dtype: {zarr_array.dtype}")
                    
                except Exception as e:
                    print(f"Error loading data from {sub_url} with fsspec/zarr: {e}")
                    raise
        else:
            # Use TensorStore for everything (both local and remote)
            if self.verbose:
                print(f"Using TensorStore for data access (use_fsspec={self.use_fsspec})")
                
            if self.cache:
                context_spec = {
                    'cache_pool': {
                        "total_bytes_limit": self.cache_pool
                    }
                }
            else:
                context_spec = {}
            
            # For each resolution level
            for dataset in self.metadata['zattrs']['multiscales'][0]['datasets']:
                path = dataset['path']
                
                try:
                    if self.domain == "local":
                        # For local files, use file kvstore driver
                        sub_url = f"{base_url}/{path}"
                        
                        if self.verbose:
                            print(f"Attempting to load local data from: {sub_url} using TensorStore")
                            
                        # Check if path exists
                        if not os.path.exists(sub_url) and os.path.exists(base_url):
                            # Try opening as a multi-resolution zarr store
                            if self.verbose:
                                print(f"Path {sub_url} not found, trying multi-resolution zarr at {base_url}")
                                
                            # This is likely a zarr hierarchy with subgroups
                            # First, open using zarr to get a list of keys
                            try:
                                import zarr
                                z = zarr.open(base_url, mode='r')
                                
                                # If path is in the keys, use that
                                if path in z.keys():
                                    sub_url = f"{base_url}/{path}"
                                else:
                                    # Use the first key
                                    key = sorted(z.keys())[0]
                                    sub_url = f"{base_url}/{key}"
                                    if self.verbose:
                                        print(f"Using first group in zarr store: {key}")
                            except Exception as e:
                                print(f"Error examining zarr store: {e}")
                                # Continue with original path
                        
                        # Create a file-based kvstore spec for local files
                        kvstore_spec = {
                            'driver': 'file',
                            'path': sub_url,
                        }
                    else:
                        # For remote data, use http kvstore driver
                        sub_url = f"{base_url}/{path}/"
                        
                        if self.verbose:
                            print(f"Attempting to load remote data from: {sub_url} using TensorStore")
                            
                        kvstore_spec = {
                            'driver': 'http',
                            'base_url': sub_url
                        }
                    
                    # Common spec for both local and remote
                    spec = {
                        'driver': 'zarr',
                        'kvstore': kvstore_spec,
                        'context': context_spec
                    }
                    
                    if self.verbose:
                        print(f"TensorStore spec: {spec}")
                    
                    # Open the TensorStore
                    data = ts.open(spec).result()
                    sub_volumes.append(data)
                    
                    if self.verbose:
                        print(f"Successfully loaded data with TensorStore")
                        print(f"Shape: {data.shape}, dtype: {data.dtype.numpy_dtype}")
                        
                except Exception as e:
                    print(f"Error loading data with TensorStore: {e}")
                    print(f"URL: {sub_url}")
                    raise

        return sub_volumes
    
    def download_inklabel(self) -> None:
        """
        Download the ink label image for a segment.

        This method downloads and sets the `inklabel` attribute for the segment.

        Raises
        ------
        AssertionError
            If the volume type is not 'segment'.
        """
        assert self.type == "segment", "Can download ink label only for segments."
        if self.url[-1] == "/":
            inklabel_url = self.url[:-6]+"_inklabels.png"
        else:
            inklabel_url = self.url[:-5]+"_inklabels.png"
        if self.domain == "local":
            # If domain is local, open the image from the local file path
            if os.path.exists(inklabel_url):
                self.inklabel = np.array(Image.open(inklabel_url))
            else:
                print(f"File not found: {inklabel_url}")
        else:
            # Make a GET request to the URL to download the image
            response = requests.get(inklabel_url)

            # Check if the request was successful
            if response.status_code == 200:
                # Open the image directly from the response content using PIL
                self.inklabel = np.array(Image.open(BytesIO(response.content)))
            else:
                print(f"Failed to download inklabel. Status code: {response.status_code}")
        

    def __getitem__(self, idx: Union[Tuple[int, ...],int]) -> NDArray:
        """
        Get a sub-volume or slice of the data.

        Parameters
        ----------
        idx : Union[Tuple[int, ...], int]
            Index tuple or integer to select the data. 
            
            For 3D data, the tuple order is (z, y, x) - this is the standard order for scientific data and TensorStore.
            
            If a fourth index is provided, it selects a specific sub-volume (resolution level),
            otherwise the data are taken from the first sub-volume (highest resolution).

        Returns
        -------
        NDArray
            The selected sub-volume or slice of data.

        Raises
        ------
        IndexError
            If the index is invalid.
        ValueError
            If the domain is invalid.
        """
        # Determine subvolume_idx and coordinates
        # Use z,y,x order to match the normal standard in scientific data and TensorStore
        if isinstance(idx, tuple) and len(idx) == 4:
            # Format is (c, z, y, x) for 4D data or (z, y, x, subvolume) for 3D with subvolume
            if self.verbose:
                print(f"Received 4D indexing: {idx}")
                
            if hasattr(self, 'data') and len(self.data) > 0 and len(self.data[0].shape) == 4:
                # This is 4D data with channels (c, z, y, x)
                z, y, x = idx[1], idx[2], idx[3]
                c = idx[0]  # Channel index
                subvolume_idx = 0
            else:
                # This is 3D data with specified subvolume (z, y, x, subvolume)
                z, y, x, subvolume_idx = idx
                
            assert 0 <= subvolume_idx < len(self.data), "Invalid subvolume index."
            
        elif isinstance(idx, tuple) and len(idx) == 3:
            # Format is (z, y, x) for standard 3D indexing
            z, y, x = idx
            subvolume_idx = 0
            
        elif isinstance(idx, tuple) and len(idx) == 2:
            # Format might be (z, y) or (c, z) depending on context
            if hasattr(self, 'data') and len(self.data) > 0 and len(self.data[0].shape) == 4:
                # This is likely 4D data, so this is (c, z)
                c, z = idx
                y = slice(None)  # All Y
                x = slice(None)  # All X
            else:
                # This is 3D data, so this is (z, y)
                z, y = idx
                x = slice(None)  # All X
            subvolume_idx = 0
            
        elif (isinstance(idx, tuple) and len(idx) == 1) or isinstance(idx, int):
            # Single index can be z or c depending on data dimensions
            if isinstance(idx, tuple):
                z = idx[0]
            else:
                z = idx
            y = slice(None)  # All Y
            x = slice(None)  # All X
            subvolume_idx = 0
            
        else:
            raise IndexError("Invalid index. Must be a tuple of one to four elements, or an integer.")
            
        # Handle different data access methods based solely on use_fsspec flag
        if self.use_fsspec:
            # Using fsspec/zarr direct indexing (for both local and remote)
            try:
                # For slice objects, just use them directly
                # Access with z,y,x ordering as this is the standard for scientific data
                if self.verbose:
                    print(f"Zarr accessing with indices: (z={z}, y={y}, x={x}), subvolume: {subvolume_idx}")
                    print(f"Data shape: {self.data[subvolume_idx].shape if subvolume_idx < len(self.data) else 'UNKNOWN'}")
                
                data_slice = self.data[subvolume_idx][z, y, x]
                
                if self.verbose:
                    print(f"Zarr returned shape: {data_slice.shape}")
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in zarr slice with indices (z={z}, y={y}, x={x}): {e}")
                    print(f"Data shape: {self.data[subvolume_idx].shape if subvolume_idx < len(self.data) else 'UNKNOWN'}")
                    print(f"Index types: z={type(z)}, y={type(y)}, x={type(x)}")
                
                # Try to work around zarr/fsspec limitations by handling different slice patterns
                if isinstance(x, slice) and isinstance(y, slice) and isinstance(z, slice):
                    # For all slice objects, we can try different approach
                    if self.verbose:
                        print(f"Trying alternative slice method for zarr with z={z}, y={y}, x={x}")
                    # Get the raw array and index it directly
                    data_slice = self.data[subvolume_idx][:]
                    data_slice = data_slice[z, y, x]
                else:
                    # Re-raise the error if we can't handle it
                    raise
                
        else:
            # Using TensorStore (for both local and remote)
            try:
                # TensorStore requires .read().result() to fetch data
                # Add debugging to understand the indexing behavior
                if self.verbose:
                    print(f"TensorStore accessing with indices: (z={z}, y={y}, x={x}), subvolume: {subvolume_idx}")
                    print(f"Data shape: {self.data[subvolume_idx].shape}")
                
                # Properly handle slices to work with TensorStore
                # TensorStore expects dimensions in z,y,x order
                data_slice = self.data[subvolume_idx][z, y, x].read().result()
                
                if self.verbose:
                    print(f"TensorStore returned shape: {data_slice.shape}")

            except Exception as e:
                if self.verbose:
                    print(f"TensorStore access error: {e}")
                    print(f"Indices: z={z}, y={y}, x={x}, type(z)={type(z)}, type(y)={type(y)}, type(x)={type(x)}")
                # Re-raise to let caller handle the error
                raise

        # we perform all normalization in float32 , and then scale for output dtypes if specified in 'return_as_type'

        # Apply normalization if needed
        if self.normalize:
            data_slice = data_slice / self.max_dtype
            data_slice = data_slice.astype(np.float32)

        if self.normalization_scheme:
            if self.normalization_scheme == 'zscore' or self.normalization_scheme == 'std':
                # zscore and std do the same normalization: (x - mean) / std
                for c in range(data_slice.shape[0]):
                    mean = np.mean(data_slice[c])
                    std = np.std(data_slice[c])
                    if std > 0:
                        data_slice[c] = (data_slice[c] - mean) / std
            elif self.normalization_scheme == 'minmax':
                for c in range(data_slice.shape[0]):
                    min_val = np.min(data_slice[c])
                    max_val = np.max(data_slice[c])
                    if max_val > min_val:
                        data_slice[c] = (data_slice[c] - min_val) / (max_val - min_val)


        if self.return_as_type:
            if self.return_as_type == 'np.float32':
                pass # data is float32 at this stage, we just pass it along
            elif self.return_as_type == 'np.uint8':
                data_slice = data_slice * get_max_value(np.uint8)
                data_slice = data_slice.astype(np.uint8)
            elif self.return_as_type == 'np.uint16':
                data_slice = data_slice * get_max_value(np.uint16)
                data_slice = data_slice.astype(np.uint16)
            elif self.return_as_type == 'np.float16':
                data_slice = data_slice * get_max_value(np.float16)
                data_slice = data_slice.astype(np.float16)

        if self.return_as_tensor:
            data_slice = torch.from_numpy(data_slice).contiguous()
            return data_slice
        else:
            return data_slice



                    

        
    def grab_canonical_energy(self) -> Optional[int]:
        """
        Get the canonical energy for the volume based on the scroll ID.

        Returns
        -------
        int
            The canonical energy value.
        """
        energy_mapping = {
            1: 54, "1b": 54, 2: 54, "2b": 54, "2c": 88,
            3: 53, 4: 88, 5: 53
        }
        return energy_mapping.get(self.scroll_id, None)

    def grab_canonical_resolution(self) -> Optional[float]:
        """
        Get the canonical resolution for the volume based on the scroll ID.

        Returns
        -------
        float
            The canonical resolution value.
        """
        resolution_mapping = {
            1: 7.91, "1b": 7.91, 2: 7.91, "2b": 7.91, "2c": 7.91,
            3: 3.24, 4: 3.24, 5: 7.91
        }
        return resolution_mapping.get(self.scroll_id, None)
                
    def activate_caching(self) -> None:
        """
        Activate caching for the volume data.

        This method enables caching and reloads the data with caching enabled.
        """
        if self.domain != "local":
            if not self.cache:
                self.cache = True
                self.data = self.load_data()

    def deactivate_caching(self) -> None:
        """
        Deactivate caching for the volume data.

        This method disables caching and reloads the data without caching.
        """
        if self.domain != "local":
            if self.cache:
                self.cache = False
                self.data = self.load_data()

    def shape(self, subvolume_idx: int = 0) -> Tuple[int, ...]:
        """
        Get the shape of a specific sub-volume.

        Parameters
        ----------
        subvolume_idx : int, default = 0
            Index of the sub-volume to get the shape of.

        Returns
        -------
        Tuple[int, ...]
            The shape of the specified sub-volume.

        Raises
        ------
        AssertionError
            If the sub-volume index is invalid.
        """
        assert 0 <= subvolume_idx < len(self.data), "Invalid subvolume index"
        return self.data[subvolume_idx].shape

  
class Cube:
    """
    A class to represent a 3D instance annotated cube within a scroll.

    Attributes
    ----------
    scroll_id : int
        ID of the scroll.
    energy : int
        Energy value associated with the cube.
    resolution : float
        Resolution of the cube.
    z : int
        Z-coordinate of the cube.
    y : int
        Y-coordinate of the cube.
    x : int
        X-coordinate of the cube.
    cache : bool
        Indicates if caching is enabled.
    cache_dir : Optional[os.PathLike]
        Directory where cached files are stored.
    normalize : bool
        Indicates if the data should be normalized.
    configs : str
        Path to the configuration file.
    volume_url : str
        URL to access the volume data.
    mask_url : str
        URL to access the mask data.
    volume : NDArray
        Loaded volume data.
    mask : NDArray
        Loaded mask data.
    max_dtype : Union[float, int]
        Maximum value of the dtype if normalization is enabled.
    """
    def __init__(self, scroll_id: int, energy: int, resolution: float, z: int, y: int, x: int, cache: bool = False, cache_dir : Optional[os.PathLike] = None, normalize: bool = False) -> None:
        """
        Initialize the Cube object.

        Parameters
        ----------
        scroll_id : int
            ID of the scroll.
        energy : int
            Energy value associated with the cube.
        resolution : float
            Resolution of the cube.
        z : int
            Z-coordinate of the cube.
        y : int
            Y-coordinate of the cube.
        x : int
            X-coordinate of the cube.
        cache : bool, default = False
            Indicates if caching is enabled.
        cache_dir : Optional[os.PathLike], default = None
            Directory where cached files are stored. If None the files will be saved in $HOME / vesuvius / annotated-instances
        normalize : bool, default = False
            Indicates if the data should be normalized.

        Raises
        ------
        ValueError
            If the URL cannot be found in the configuration.
        """
        self.scroll_id = scroll_id
        install_path = get_installation_path()
        
        # Try different possible locations for the config file
        possible_paths = [
            os.path.join(install_path, 'setup', 'configs', f'cubes.yaml'),  # For editable installs
            os.path.join(install_path, 'vesuvius', 'setup', 'configs', f'cubes.yaml'),  # For regular installs
            os.path.join(install_path, 'configs', f'cubes.yaml')  # Fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.configs = path
                break
        else:
            # If no path exists, use the first one and we'll handle the error later
            self.configs = possible_paths[0]
        self.energy = energy
        self.resolution = resolution
        self.z, self.y, self.x = z, y, x
        self.volume_url, self.mask_url = self.get_url_from_yaml()
        self.aws = is_aws_ec2_instance()
        if self.aws is False:
            self.cache = cache
            if self.cache:
                if cache_dir is not None:
                    self.cache_dir = Path(cache_dir)
                else:
                    self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'
                os.makedirs(self.cache_dir, exist_ok=True)
        self.normalize = normalize

        self.volume, self.mask = self.load_data()

        if self.normalize:
            self.max_dtype = get_max_value(self.volume.dtype)
        
    def get_url_from_yaml(self) -> str:
        """
        Retrieve the URLs for the volume and mask data from the YAML configuration file.

        Returns
        -------
        Tuple[str, str]
            The URLs for the volume and mask data.

        Raises
        ------
        ValueError
            If the URLs cannot be found in the configuration.
        FileNotFoundError
            If the configuration file doesn't exist.
        """
        try:
            # Load the YAML file
            with open(self.configs, 'r') as file:
                data: Dict[str, Any] = yaml.safe_load(file)
            
            # Handle empty file or invalid YAML
            if not data:
                error_msg = f"Config file at {self.configs} is empty or invalid. "
                error_msg += f"You need to populate it with data for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, cube: {self.z:05d}_{self.y:05d}_{self.x:05d}"
                raise ValueError(error_msg)
                
            # Retrieve the URL for the given id, energy, and resolution
            base_url: str = data.get(self.scroll_id, {}).get(self.energy, {}).get(self.resolution, {}).get(f"{self.z:05d}_{self.y:05d}_{self.x:05d}")
            if base_url is None:
                raise ValueError(f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, cube: {self.z:05d}_{self.y:05d}_{self.x:05d}. Make sure these values are in your config file.")

            volume_filename = f"{self.z:05d}_{self.y:05d}_{self.x:05d}_volume.nrrd"
            mask_filename = f"{self.z:05d}_{self.y:05d}_{self.x:05d}_mask.nrrd"

            volume_url = os.path.join(base_url, volume_filename)
            mask_url = os.path.join(base_url, mask_filename)
            return volume_url, mask_url
            
        except FileNotFoundError:
            error_msg = f"Configuration file not found at {self.configs}. "
            error_msg += "Please make sure the vesuvius package is properly installed with configuration files.\n"
            error_msg += "You can download example config files from: https://github.com/ScrollPrize/villa/tree/main/setup/configs"
            raise FileNotFoundError(error_msg)
    
    def load_data(self) -> Tuple[NDArray, NDArray]:
        """
        Load the data for the cube.

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing the loaded volume and mask data.

        Raises
        ------
        requests.RequestException
            If there is an error downloading the data from the server.
        """
        output = []
        for url in [self.volume_url, self.mask_url]:
            if self.aws:
                array, _ = nrrd.read(url)
            else:
                if self.cache:
                    # Extract the relevant path after "instance-labels"
                    path_after_finished_cubes = url.split('instance-labels/')[1]
                    # Extract the directory structure and the filename
                    dir_structure, filename = os.path.split(path_after_finished_cubes)

                    # Create the full directory path in the temp_dir
                    full_temp_dir_path = os.path.join(self.cache_dir, dir_structure)

                    # Make sure the directory structure exists
                    os.makedirs(full_temp_dir_path, exist_ok=True)

                    # Create the full path for the temporary file
                    temp_file_path = os.path.join(full_temp_dir_path, filename)

                    # Check if the file already exists in the cache
                    if os.path.exists(temp_file_path):
                        # Read the NRRD file from the cache
                        array, _ = nrrd.read(temp_file_path)

                    else:
                        # Download the remote file
                        response = requests.get(url)
                        response.raise_for_status()  # Ensure we notice bad responses
                        # Write the downloaded content to the temporary file with the same directory structure and filename
                        with open(temp_file_path, 'wb') as tmp_file:
                            tmp_file.write(response.content)

                            array, _ = nrrd.read(temp_file_path)

                else:
                    response = requests.get(url)
                    response.raise_for_status()  # Ensure we notice bad responses
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        temp_file_path = tmp_file.name
                        # Read the NRRD file from the temporary file

                        array, _ = nrrd.read(temp_file_path)

            output.append(array)

        return output[0], output[1]


    def __getitem__(self, idx: Tuple[int, ...]) -> NDArray:
        """
        Get a slice of the cube data.

        Parameters
        ----------
        idx : Tuple[int, ...]
            A tuple representing the coordinates (z, y, x) within the cube.

        Returns
        -------
        NDArray
            The selected data slice.

        Raises
        ------
        IndexError
            If the index is invalid.
        """
        if isinstance(idx, tuple) and len(idx) == 3:
            zz, yy, xx = idx

            if self.normalize:
                return self.volume[zz, yy, xx]/self.max_dtype, self.mask[zz, yy, xx]
            
            else:
                return self.volume[zz, yy, xx], self.mask[zz, yy, xx]
            
        else:
            raise IndexError("Invalid index. Must be a tuple of three elements.")
    
    def activate_caching(self, cache_dir: Optional[os.PathLike] = None) -> None:
        """
        Activate caching for the cube data.

        Parameters
        ----------
        cache_dir : Optional[os.PathLike], default = None
            Directory where cached files are stored.
        """
        if not self.cache:
            if cache_dir is None:
                self.cache_dir = Path.home() / 'vesuvius' / 'annotated-instances'
            else:
                self.cache_dir = Path(cache_dir)
            self.cache = True
            self.volume, self.mask = self.load_data()

    def deactivate_caching(self) -> None:
        """
        Deactivate caching for the cube data.
        """
        if self.cache:
            self.cache = False
            self.volume, self.mask = self.load_data()

