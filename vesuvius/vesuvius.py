"""
Vesuvius module - provides access to the package components.
"""

# Import key modules
import data
import models
import utils
import setup

# Import specific classes for direct access
from data.volume import Volume, Cube
from data.vc_dataset import VCDataset

# Define what to expose
__all__ = ['data', 'models', 'utils', 'setup', 'Volume', 'Cube', 'VCDataset']