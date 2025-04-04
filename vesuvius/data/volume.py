import os
import yaml
import json
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
# Direct import to avoid circular reference issues
from data.io.paths import list_files, is_aws_ec2_instance
import torch
import tensorstore as ts
from .utils import get_max_value

# Remove the PIL image size limit
Image.MAX_IMAGE_PIXELS = None

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
        Size of the cache pool in bytes.
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
        Loaded volume data using TensorStore.
    inklabel : np.ndarray
        Ink label data (only for segments).
    dtype : np.dtype
        Data type of the volume.
    """

    def __init__(self, type: Union[str, int],
                 scroll_id: Optional[Union[int, str]] = None,
                 energy: Optional[int] = None,
                 resolution: Optional[float] = None,
                 segment_id: Optional[int] = None,
                 cache: bool = True, cache_pool: int = 1e10,
                 format: str = 'zarr',
                 normalize: bool = False,
                 normalization_scheme: str = 'none',
                 return_as_type: str = 'none',  # none in this parameter indicates no dtype conversion will occur
                 return_as_tensor: bool = False,
                 verbose: bool = False,
                 domain: Optional[str] = None,
                 path: Optional[str] = None,
                 ):

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
        format : str, default = 'zarr'
            Format of the data store. Currently only 'zarr' is supported.
        normalize : bool, default = False
            Indicates if the data should be normalized to float  (values between 0 and 1)
        normalization_scheme : str, default = 'none'
            In addition to normalizing to float values, data can be additionally normalized by performing one of the normalization schemes,
            available options are 'zscore'/'std' and 'minmax'
        return_as_type : str, default = 'none'
            Specify the type of data you'd like returned, options are 'np.uint8', 'np.uint16', 'np.float16', 'np.float32'. 
            'none' will simply return in the original dtype.
        return_as_tensor: bool, default = False
            If True, returns the data as a PyTorch tensor instead of a NumPy array
        verbose : bool, default = False
            If True, prints additional information during initialization.
        domain : str, default = "dl.ash2txt"
            The domain from where data is fetched: 'dl.ash2txt' or 'local'.
        path : Optional[str], default = None
            Path to the local data if domain is 'local'.

        Raises
        ------
        ValueError
            If the provided `type` or `domain` is invalid.
        Exception
            If there's an error opening or accessing the data store.
        """

        # Initialize basic attributes
        self.format = format
        self.cache = cache
        self.cache_pool = cache_pool
        self.normalize = normalize
        self.normalization_scheme = normalization_scheme
        self.return_as_type = return_as_type
        self.return_as_tensor = return_as_tensor
        self.path = path
        self.verbose = verbose

        try:
            if format == "zarr" and self.path is not None:
                # Configure TensorStore for direct zarr access
                cache_pool_bytes = int(self.cache_pool) if self.cache else 0
                
                # Determine if the path is HTTP/HTTPS or local
                is_http = self.path.startswith(('http://', 'https://'))
                
                # Configure the kvstore driver based on path type
                if is_http:
                    kvstore_config = {
                        'driver': 'http',
                        'base_url': self.path,
                    }
                else:
                    kvstore_config = {
                        'driver': 'file',
                        'path': self.path,
                    }
                
                ts_config = {
                    'driver': 'zarr',
                    'kvstore': kvstore_config,
                    'context': {
                        'cache_pool': {
                            'total_bytes_limit': cache_pool_bytes
                        }
                    }
                }
                
                if self.verbose:
                    print(f"Opening zarr store with TensorStore at path: {self.path}")
                    print(f"TensorStore config: {json.dumps(ts_config, indent=2)}")
                
                try:
                    # Set type and other required attributes first
                    self.type = "zarr"
                    self.scroll_id = None
                    self.segment_id = None
                    self.domain = "local" if not is_http else "dl.ash2txt"
                    
                    # Fix URL format - remove trailing slash if present for consistency
                    self.url = self.path.rstrip("/")
                    
                    # Set the resolution parameter
                    self.resolution = 0.0
                    
                    # First try to open directly - this will work if .zarray is at the root level
                    try:
                        if self.verbose:
                            print(f"Attempting to open zarr store directly at: {self.path}")
                        future = ts.open(ts_config)
                        self.data = [future.result()]
                        if self.verbose:
                            print(f"Successfully opened zarr store at: {self.path}")
                    except Exception as root_e:
                        if self.verbose:
                            print(f"Error opening zarr at root level: {root_e}")
                        
                        # If that fails, try to open with "/0" appended for multiresolution zarr stores
                        # where .zarray is in subgroups
                        try:
                            # Make sure we don't get a double-slash in the URL
                            subpath = os.path.join(self.url, "0")
                            if self.verbose:
                                print(f"Attempting to open multiresolution zarr store at: {subpath}")
                            
                            # Modify the config to add path="/0" for zarr driver
                            multi_ts_config = ts_config.copy()
                            multi_ts_config['path'] = "0"
                            
                            future = ts.open(multi_ts_config)
                            result = future.result()
                            self.data = [result]
                            
                            # Check if this is truly a multiresolution zarr by trying to open /1, /2, etc.
                            try:
                                resolutions = []
                                # Start with the first resolution (already loaded)
                                resolutions.append(result)
                                
                                # Try to load additional resolutions
                                for res_level in [1, 2, 3, 4, 5]:  # Try up to 5 resolution levels
                                    res_config = ts_config.copy()
                                    res_config['path'] = str(res_level)
                                    try:
                                        if self.verbose:
                                            print(f"Trying to open resolution level {res_level}")
                                        res_future = ts.open(res_config)
                                        res_result = res_future.result()
                                        resolutions.append(res_result)
                                        if self.verbose:
                                            print(f"Successfully loaded resolution level {res_level}")
                                    except Exception as res_e:
                                        if self.verbose:
                                            print(f"No more resolution levels after {res_level-1}: {res_e}")
                                        break
                                
                                # If we found multiple resolutions, update self.data
                                if len(resolutions) > 1:
                                    self.data = resolutions
                                    if self.verbose:
                                        print(f"Loaded {len(resolutions)} resolution levels")
                            except Exception as multi_e:
                                if self.verbose:
                                    print(f"Error checking for additional resolution levels: {multi_e}")
                                # Continue with just the first resolution level
                                pass
                            
                            if self.verbose:
                                print(f"Successfully opened multiresolution zarr store")
                        except Exception as subgroup_e:
                            # Re-raise the original error if all attempts fail
                            if self.verbose:
                                print(f"Error opening zarr with subgroup path: {subgroup_e}")
                            raise root_e
                    
                    # Get dtype from TensorStore
                    self.dtype = self.data[0].dtype.numpy_dtype
                    
                    if self.normalize:
                        self.max_dtype = get_max_value(self.dtype)
                    
                    # Try to load the actual metadata from the zarr store
                    if is_http:
                        # For HTTP zarr, use requests to get the metadata
                        try:
                            zattrs_url = f"{self.path.rstrip('/')}/.zattrs"
                            response = requests.get(zattrs_url)
                            response.raise_for_status()
                            self.metadata = {"zattrs": response.json()}
                            
                            if self.verbose:
                                print(f"Loaded metadata from {zattrs_url}")
                        except Exception as e:
                            # If root .zattrs fails, try in the "0" subgroup
                            try:
                                subgroup_url = f"{self.path.rstrip('/')}/0/.zattrs"
                                if self.verbose:
                                    print(f"Root .zattrs not found, trying {subgroup_url}")
                                response = requests.get(subgroup_url)
                                response.raise_for_status()
                                self.metadata = {"zattrs": response.json()}
                                
                                if self.verbose:
                                    print(f"Loaded metadata from {subgroup_url}")
                            except Exception as sub_e:
                                if self.verbose:
                                    print(f"Failed to load metadata: {sub_e}")
                                raise ValueError(f"Could not load .zattrs metadata from {self.path}")
                    else:
                        # For local zarr, check the filesystem
                        try:
                            zattrs_path = os.path.join(self.path, ".zattrs")
                            if os.path.exists(zattrs_path):
                                with open(zattrs_path, 'r') as f:
                                    self.metadata = {"zattrs": json.load(f)}
                                
                                if self.verbose:
                                    print(f"Loaded metadata from {zattrs_path}")
                            else:
                                # If root .zattrs doesn't exist, try in the "0" subgroup
                                subgroup_path = os.path.join(self.path, "0", ".zattrs")
                                if os.path.exists(subgroup_path):
                                    if self.verbose:
                                        print(f"Root .zattrs not found, trying {subgroup_path}")
                                    with open(subgroup_path, 'r') as f:
                                        self.metadata = {"zattrs": json.load(f)}
                                    
                                    if self.verbose:
                                        print(f"Loaded metadata from {subgroup_path}")
                                else:
                                    if self.verbose:
                                        print(f"Could not find .zattrs at {zattrs_path} or {subgroup_path}")
                                    raise ValueError(f"Could not find .zattrs metadata in {self.path}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Failed to load metadata: {e}")
                            raise
                    
                    # Skip the rest of initialization as we already have the data
                    if self.verbose:
                        self.meta()
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"Error opening zarr with TensorStore: {e}")
                    # Continue with initialization in case there's another way to open this data

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

            # Use relative paths for config files instead of installation path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Try different possible locations for the config file
            possible_paths = [
                os.path.join(base_dir, 'setup', 'configs', f'scrolls.yaml'),  # For editable installs
                os.path.join(base_dir, 'configs', f'scrolls.yaml')  # Fallback
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

            # Now that self.configs is set, we can check for zarr type using config file
            if self.type == 'zarr' and self.path == self.configs:
                raise ValueError(f"Invalid zarr path: Config file ({self.configs}) cannot be used as zarr store")

            if energy:
                self.energy = energy
            else:
                self.energy = self.grab_canonical_energy()

            if resolution:
                self.resolution = resolution
            else:
                self.resolution = self.grab_canonical_resolution()

            self.domain = domain

            if self.domain == "dl.ash2txt":
                # For remote paths, get the URL from the config
                self.url = self.get_url_from_yaml()
                if self.verbose:
                    print(f"Using URL from config: {self.url}")

                # Load remote data and metadata using TensorStore
                self.metadata = self.load_ome_metadata()
                self.data = self.load_data()

                # Handle dtype for TensorStore data
                if self.normalize:
                    self.max_dtype = get_max_value(self.data[0].dtype.numpy_dtype)
                self.dtype = self.data[0].dtype.numpy_dtype

            elif self.domain == "local":
                if self.verbose:
                    print(f"Opening local zarr store at: {self.url}")

                # Load local data using TensorStore
                self.metadata = self.load_ome_metadata()
                self.data = self.load_data()
                
                # Handle dtype for TensorStore data
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
            print('Load the canonical scroll 1 with Volume(type="scroll", scroll_id=1, energy=54, resolution=7.91)',
                  end="\n")
            print(
                'If loading another part of the same physical scroll use for instance Volume(type="scroll", scroll_id="1b", energy=54, resolution=7.91)',
                end="\n")
            print(
                'Load a segment (e.g. 20230827161847) with Volume(type="segment", scroll_id=1, energy=54, resolution=7.91, segment_id=20230827161847)')
            raise

    def meta(self) -> None:
        """
        Print metadata information about the volume.

        This method provides information about the resolution and shape of the data at the original and scaled resolutions.
        """
        # Print just the shape information for each subvolume
        for idx in range(len(self.data)):
            print(f"Subvolume {idx} shape: {self.shape(idx)}")

    def find_segment_details(self, segment_id: str) -> Tuple[
        Optional[int], Optional[int], Optional[float], Optional[Dict[str, Any]]]:
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
                print(f"Config file path: {self.configs}")
            
            # Ensure the path isn't the same as the config file path
            if self.path == self.configs:
                raise ValueError(f"Invalid zarr path: Configuration file path '{self.path}' cannot be used as zarr store")
                
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
                url: str = data.get(str(self.scroll_id), {}).get(str(self.energy), {}).get(str(self.resolution),
                                                                                           {}).get("volume")
            elif self.type == 'segment':
                url: str = data.get(str(self.scroll_id), {}).get(str(self.energy), {}).get(str(self.resolution),
                                                                                           {}).get("segments", {}).get(
                    str(self.segment_id))
            else:
                raise ValueError(f"Cannot retrieve URL from config for type: {self.type}")

            if url is None:
                if self.type == 'scroll':
                    raise ValueError(
                        f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}. Make sure these values are in your config file.")
                elif self.type == 'segment':
                    raise ValueError(
                        f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, segment: {self.segment_id}. Make sure these values are in your config file.")
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
            # Fix URL format - remove trailing slash if present
            base_url = self.url.rstrip("/")
            
            if self.domain == "dl.ash2txt" or base_url.startswith(('http://', 'https://')):
                # Try root level .zattrs first
                zattrs_url = f"{base_url}/.zattrs"

                if self.verbose:
                    print(f"Attempting to load metadata from: {zattrs_url}")

                try:
                    # Use direct HTTP requests to load metadata from remote
                    response = requests.get(zattrs_url)
                    response.raise_for_status()
                    zattrs = response.json()
                except Exception as root_e:
                    if self.verbose:
                        print(f"Error loading root metadata, trying subgroup /0: {root_e}")
                    
                    # If that fails, try in the /0 subgroup for multiresolution zarrs
                    subgroup_zattrs_url = f"{base_url}/0/.zattrs"
                    if self.verbose:
                        print(f"Attempting to load metadata from subgroup: {subgroup_zattrs_url}")
                    
                    response = requests.get(subgroup_zattrs_url)
                    response.raise_for_status()
                    zattrs = response.json()

            elif self.domain == "local":
                # For local domain, try root level .zattrs first
                zattrs_path = os.path.join(base_url, '.zattrs')
                
                if os.path.exists(zattrs_path):
                    if self.verbose:
                        print(f"Reading metadata from: {zattrs_path}")
                    
                    # Load JSON metadata directly from the file
                    with open(zattrs_path, 'r') as f:
                        zattrs = json.load(f)
                else:
                    # If not found, check for multiresolution zarr with .zattrs in /0
                    subgroup_zattrs_path = os.path.join(base_url, "0", ".zattrs")
                    
                    if self.verbose:
                        print(f"Root .zattrs not found, trying: {subgroup_zattrs_path}")
                    
                    if os.path.exists(subgroup_zattrs_path):
                        with open(subgroup_zattrs_path, 'r') as f:
                            zattrs = json.load(f)
                    else:
                        raise FileNotFoundError(f"Could not find .zattrs at {zattrs_path} or {subgroup_zattrs_path}")

            return {
                "zattrs": zattrs,
            }
        except Exception as e:
            print(f"Error loading metadata: {e}")
            raise

    def load_data(self):
        """
        Load the data for the volume using TensorStore.

        Returns
        -------
        List[ts.TensorStore]
            A list of TensorStore objects representing the sub-volumes.

        Raises
        ------
        Exception
            If there is an error loading the data from the server.
        """
        # Fix URL format - remove trailing slash if present
        base_url = self.url.rstrip("/")
        sub_volumes = []

        # For each resolution level
        for dataset in self.metadata['zattrs']['multiscales'][0]['datasets']:
            path = dataset['path']
            sub_url = f"{base_url}/{path}"

            if self.verbose:
                print(f"Attempting to load data from: {sub_url} using TensorStore")

            try:
                # Configure cache pool size based on user settings
                cache_pool_bytes = int(self.cache_pool) if self.cache else 0
                
                # Common TensorStore context with cache settings
                context = {
                    'cache_pool': {
                        'total_bytes_limit': cache_pool_bytes
                    }
                }
                
                if self.domain == "local":
                    # Local file system access
                    kvstore_config = {
                        'driver': 'file',
                        'path': base_url,  # Use base URL as the path for all resolutions
                    }
                    
                    # Check for multiresolution zarr
                    multiresolution = False
                    zarray_path = os.path.join(sub_url, '.zarray')
                    base_zarray_path = os.path.join(base_url, '.zarray')
                    
                    # First check if this is a traditional zarr with .zarray at the path level
                    if os.path.exists(zarray_path):
                        # This is a standard zarr with .zarray at the sub_url level
                        kvstore_config = {
                            'driver': 'file',
                            'path': sub_url,
                        }
                    # Then check if this is a multiresolution zarr with .zarray in numbered groups
                    elif os.path.exists(os.path.join(base_url, path, '.zarray')):
                        # This is a multiresolution zarr with .zarray in numbered groups
                        multiresolution = True
                    # Otherwise check if .zarray is at the base URL
                    elif os.path.exists(base_zarray_path):
                        # This is a standard zarr with .zarray at the root
                        pass
                    else:
                        raise FileNotFoundError(f"Cannot find zarr data at {sub_url}, {base_url}, or {os.path.join(base_url, path)}")
                
                else:
                    # Remote HTTP access
                    if not base_url.startswith(('http://', 'https://')):
                        raise ValueError(f"Invalid URL for HTTP access: {base_url}")
                    
                    # For remote paths, assume multiresolution by default and try both methods
                    kvstore_config = {
                        'driver': 'http',
                        'base_url': base_url,  # Use base URL as the base for all resolutions
                    }
                    multiresolution = True
                    
                    # Validate the sub_url
                    if not sub_url.startswith(('http://', 'https://')):
                        # If somehow sub_url doesn't have the protocol, it's an error
                        raise ValueError(f"Invalid URL for HTTP access: {sub_url}")
                
                # Create and open TensorStore
                ts_config = {
                    'driver': 'zarr',
                    'kvstore': kvstore_config,
                    'context': context,
                }
                
                # For multiresolution zarrs, add the path to the zarr config
                if multiresolution:
                    ts_config['path'] = path
                
                if self.verbose:
                    print(f"TensorStore config: {json.dumps(ts_config, indent=2)}")
                
                # Try to open with the current configuration
                try:
                    # Open the TensorStore
                    future = ts.open(ts_config)
                    store = future.result()
                except Exception as e1:
                    if self.verbose:
                        print(f"First attempt failed: {e1}")
                    
                    # If that fails and we're using HTTP, try with the sub_url directly
                    if self.domain != "local":
                        try:
                            # Try with direct sub_url access
                            alt_config = {
                                'driver': 'zarr',
                                'kvstore': {
                                    'driver': 'http',
                                    'base_url': sub_url,
                                },
                                'context': context,
                            }
                            
                            if self.verbose:
                                print(f"Trying alternative config with direct sub_url: {json.dumps(alt_config, indent=2)}")
                                
                            future = ts.open(alt_config)
                            store = future.result()
                        except Exception as e2:
                            if self.verbose:
                                print(f"Alternative attempt also failed: {e2}")
                            # If both methods fail, raise the original error
                            raise e1
                
                sub_volumes.append(store)

                if self.verbose:
                    print(f"Successfully loaded data from {sub_url} using TensorStore")
                    print(f"Shape: {store.shape}, dtype: {store.dtype}")

            except Exception as e:
                print(f"Error loading data from {sub_url} with TensorStore: {e}")
                if len(sub_volumes) > 0:
                    # If we've successfully loaded at least one resolution level, continue with what we have
                    print(f"Continuing with {len(sub_volumes)} successfully loaded resolution levels")
                    break
                else:
                    # If we couldn't load any resolution levels, it's a critical error
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
            inklabel_url = self.url[:-6] + "_inklabels.png"
        else:
            inklabel_url = self.url[:-5] + "_inklabels.png"
            
        if self.verbose:
            print(f"Attempting to load ink label from: {inklabel_url}")
            
        # Configure TensorStore for image access
        if self.domain == "local":
            # Local file system access
            ts_config = {
                'driver': 'image',
                'kvstore': {
                    'driver': 'file',
                    'path': inklabel_url,
                }
            }
        else:
            # Remote HTTP access
            ts_config = {
                'driver': 'image',
                'kvstore': {
                    'driver': 'http',
                    'url': inklabel_url,
                }
            }
            
        try:
            # Open and read the image with TensorStore
            future = ts.open(ts_config)
            img_store = future.result()
            img_data = img_store.read().result()
            self.inklabel = np.array(img_data)
            
            if self.verbose:
                print(f"Successfully loaded ink label with shape: {self.inklabel.shape}")
        except Exception as e:
            print(f"Error loading ink label: {e}")
            # Initialize an empty ink label array with the expected shape
            if hasattr(self, 'data') and len(self.data) > 0:
                # Get x and y dimensions from the data shape
                shape = self.shape(0)
                if len(shape) >= 3:
                    # For 3D data, ink label is typically a 2D mask matching the XY dimensions
                    self.inklabel = np.zeros((shape[-2], shape[-1]), dtype=np.uint8)
                    if self.verbose:
                        print(f"Created empty ink label with shape: {self.inklabel.shape}")

    def __getitem__(self, idx: Union[Tuple[int, ...], int]) -> NDArray:
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

        try:
            # For slice objects, just use them directly with TensorStore
            # Access with z,y,x ordering as this is the standard for scientific data
            if self.verbose:
                print(f"TensorStore accessing with indices: (z={z}, y={y}, x={x}), subvolume: {subvolume_idx}")
                print(f"Data shape: {self.data[subvolume_idx].shape}")

            # TensorStore supports standard NumPy-like indexing
            # Read data from TensorStore
            future = self.data[subvolume_idx][z, y, x].read()
            data_slice = future.result()
            
            # Convert to numpy array
            data_slice = np.array(data_slice)

            if self.verbose:
                print(f"TensorStore returned shape: {data_slice.shape}")

        except Exception as e:
            if self.verbose:
                print(f"Error in TensorStore slice with indices (z={z}, y={y}, x={x}): {e}")
                print(f"Data shape: {self.data[subvolume_idx].shape}")
                print(f"Index types: z={type(z)}, y={type(y)}, x={type(x)}")
            raise

        # Perform normalization only once, consistently
        
        # Convert to float32 first if normalization is needed
        if self.normalize or self.normalization_scheme != 'none':
            data_slice = data_slice.astype(np.float32)
            
        # Apply normalization based on strategy
        if self.normalization_scheme == 'zscore' or self.normalization_scheme == 'std':
            # Z-score normalization exactly as in nnUNet's ZScoreNormalization
            for c in range(data_slice.shape[0]):
                mean = np.mean(data_slice[c])
                std = np.std(data_slice[c])
                # Use max(std, 1e-8) to match nnUNet's epsilon handling
                data_slice[c] = (data_slice[c] - mean) / (max(std, 1e-8))
        elif self.normalization_scheme == 'minmax':
            # Min-max normalization matching nnUNet's RescaleTo01Normalization 
            for c in range(data_slice.shape[0]):
                min_val = np.min(data_slice[c])
                # First shift to make minimum 0
                data_slice[c] = data_slice[c] - min_val
                # Then scale to max of 1, with epsilon to prevent division by zero
                max_val = np.max(data_slice[c])
                data_slice[c] = data_slice[c] / max(max_val, 1e-8)
        # Scale to [0,1] range if normalize=True and no other normalization applied
        elif self.normalize:
            # For TensorStore, dtype is an object, so we need to get numpy_dtype
            if hasattr(self, 'max_dtype'):
                data_slice = data_slice / self.max_dtype
            else:
                # If max_dtype is not set, calculate it
                if hasattr(self.data[subvolume_idx].dtype, 'numpy_dtype'):
                    self.max_dtype = get_max_value(self.data[subvolume_idx].dtype.numpy_dtype)
                else:
                    self.max_dtype = get_max_value(self.data[subvolume_idx].dtype)
                data_slice = data_slice / self.max_dtype

        if self.return_as_type:
            if self.return_as_type == 'np.float32':
                pass  # data is float32 at this stage, we just pass it along
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
                # Reload data with TensorStore caching enabled
                self.data = self.load_data()
                # Update dtype info for the reloaded data
                if hasattr(self.data[0].dtype, 'numpy_dtype'):
                    self.dtype = self.data[0].dtype.numpy_dtype
                else:
                    self.dtype = self.data[0].dtype
                
                if self.normalize:
                    self.max_dtype = get_max_value(self.dtype)

    def deactivate_caching(self) -> None:
        """
        Deactivate caching for the volume data.

        This method disables caching and reloads the data without caching.
        """
        if self.domain != "local":
            if self.cache:
                self.cache = False
                # Reload data with TensorStore caching disabled
                self.data = self.load_data()
                # Update dtype info for the reloaded data
                if hasattr(self.data[0].dtype, 'numpy_dtype'):
                    self.dtype = self.data[0].dtype.numpy_dtype
                else:
                    self.dtype = self.data[0].dtype
                
                if self.normalize:
                    self.max_dtype = get_max_value(self.dtype)

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
        # TensorStore stores shape as a property, return it as a tuple
        return tuple(self.data[subvolume_idx].shape)


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

    def __init__(self, scroll_id: int, energy: int, resolution: float, z: int, y: int, x: int, cache: bool = False,
                 cache_dir: Optional[os.PathLike] = None, normalize: bool = False) -> None:
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
        
        # Use relative paths for config files
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Try different possible locations for the config file
        possible_paths = [
            os.path.join(base_dir, 'setup', 'configs', f'cubes.yaml'),  # For editable installs
            os.path.join(base_dir, 'configs', f'cubes.yaml')  # Fallback
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
            base_url: str = data.get(self.scroll_id, {}).get(self.energy, {}).get(self.resolution, {}).get(
                f"{self.z:05d}_{self.y:05d}_{self.x:05d}")
            if base_url is None:
                raise ValueError(
                    f"URL not found for scroll: {self.scroll_id}, energy: {self.energy}, resolution: {self.resolution}, cube: {self.z:05d}_{self.y:05d}_{self.x:05d}. Make sure these values are in your config file.")

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
                return self.volume[zz, yy, xx] / self.max_dtype, self.mask[zz, yy, xx]

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
