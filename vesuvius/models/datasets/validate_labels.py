#!/usr/bin/env python3
"""
Script to validate label configurations and analyze dataset labels.

Usage:
    python validate_labels.py --config path/to/config.yaml --dataset napari
    python validate_labels.py --config path/to/config.yaml --dataset tif --data-path /path/to/data
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import vesuvius modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from vesuvius.models.configuration.config_manager import ConfigManager
from vesuvius.models.datasets.napari_dataset import NapariDataset
from vesuvius.models.datasets.tif_dataset import TifDataset
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.datasets.label_config_utils import analyze_dataset_labels, validate_label_data


def main():
    parser = argparse.ArgumentParser(description="Validate label configurations for vesuvius datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    parser.add_argument("--dataset", type=str, choices=["napari", "tif", "zarr"], required=True, 
                        help="Dataset type to validate")
    parser.add_argument("--data-path", type=str, help="Path to data directory (required for tif/zarr datasets)")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of patches to sample for analysis")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    mgr = ConfigManager(verbose=args.verbose)
    mgr.load_config(args.config)
    
    # Add data path if provided
    if args.data_path:
        mgr.data_path = args.data_path
    
    # Create dataset
    print(f"\nCreating {args.dataset} dataset...")
    try:
        if args.dataset == "napari":
            dataset = NapariDataset(mgr)
        elif args.dataset == "tif":
            if not args.data_path:
                raise ValueError("--data-path is required for TIF datasets")
            dataset = TifDataset(mgr)
        elif args.dataset == "zarr":
            if not args.data_path:
                raise ValueError("--data-path is required for Zarr datasets")
            dataset = ZarrDataset(mgr)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return 1
    
    print(f"Dataset created successfully with {len(dataset)} patches")
    
    # Analyze dataset labels
    print(f"\nAnalyzing labels from {args.sample_size} sample patches...")
    analysis = analyze_dataset_labels(dataset, sample_size=args.sample_size)
    
    # Print analysis results
    print("\n" + "="*60)
    print("LABEL ANALYSIS RESULTS")
    print("="*60)
    
    for target_name, target_analysis in analysis.items():
        print(f"\nTarget: {target_name}")
        print("-" * 30)
        print(f"Unique label values: {target_analysis['all_unique_values']}")
        print(f"Number of unique values: {target_analysis['num_unique_values']}")
        
        if 'validation_warning' in target_analysis:
            print(f"⚠️  WARNING: {target_analysis['validation_warning']}")
        
        # Get target configuration
        target_config = None
        if hasattr(dataset, 'target_value') and isinstance(dataset.target_value, dict):
            target_config = dataset.target_value.get(target_name)
        elif hasattr(dataset, 'target_value') and isinstance(dataset.target_value, int):
            target_config = dataset.target_value
        
        if target_config:
            print(f"Configured mapping: {target_config}")
        else:
            print("No target_value configuration found for this target")
        
        if args.verbose:
            print("\nSample patches:")
            for sample in target_analysis['samples'][:3]:  # Show first 3 samples
                print(f"  Patch {sample['patch_idx']}: values={sample['unique_values']}, shape={sample['shape']}")
    
    # Configuration recommendations
    print("\n" + "="*60)
    print("CONFIGURATION RECOMMENDATIONS")
    print("="*60)
    
    config_suggestions = []
    
    for target_name, target_analysis in analysis.items():
        unique_vals = target_analysis['all_unique_values']
        
        if len(unique_vals) == 2 and 0 in unique_vals and 1 in unique_vals:
            # Already binary
            config_suggestions.append(f"  {target_name}: 1  # Already binary (0, 1)")
        elif len(unique_vals) == 2 and 0 in unique_vals:
            # Binary with different foreground value
            fg_val = [v for v in unique_vals if v != 0][0]
            config_suggestions.append(f"  {target_name}: 1  # Binary task, original foreground value: {fg_val}")
        elif len(unique_vals) > 2:
            # Multi-class
            config_suggestions.append(f"  {target_name}:  # Multi-class task")
            for val in unique_vals:
                config_suggestions.append(f"    {val}: {val}  # Keep class {val}")
    
    if config_suggestions:
        print("\nSuggested target_value configuration:")
        print("```yaml")
        print("dataset_config:")
        print("  binarize_labels: true")
        print("  target_value:")
        for suggestion in config_suggestions:
            print(suggestion)
        print("```")
    
    # Check if current configuration matches the data
    print("\n" + "="*60)
    print("CONFIGURATION VALIDATION")
    print("="*60)
    
    all_valid = True
    
    for target_name in analysis.keys():
        target_config = None
        if hasattr(dataset, 'target_value') and isinstance(dataset.target_value, dict):
            target_config = dataset.target_value.get(target_name)
        elif hasattr(dataset, 'target_value') and isinstance(dataset.target_value, int):
            target_config = dataset.target_value
        
        if target_config is None:
            print(f"\n❌ Target '{target_name}': No configuration found")
            print(f"   Found label values: {analysis[target_name]['all_unique_values']}")
            all_valid = False
        else:
            unique_vals = analysis[target_name]['all_unique_values']
            
            if isinstance(target_config, int):
                # Binary task
                expected = [0, target_config]
                if set(unique_vals) == {0, 1} or (len(unique_vals) == 2 and 0 in unique_vals):
                    print(f"\n✅ Target '{target_name}': Valid binary configuration")
                else:
                    print(f"\n❌ Target '{target_name}': Configuration mismatch")
                    print(f"   Expected binary task but found values: {unique_vals}")
                    all_valid = False
            elif isinstance(target_config, dict):
                # Multi-class task
                configured_values = set(target_config.keys())
                label_values = set(v for v in unique_vals if v > 0)
                missing = label_values - configured_values
                
                if missing:
                    print(f"\n❌ Target '{target_name}': Missing class mappings")
                    print(f"   Values {sorted(missing)} found in data but not in configuration")
                    all_valid = False
                else:
                    print(f"\n✅ Target '{target_name}': Valid multi-class configuration")
    
    if all_valid:
        print("\n✅ All configurations are valid!")
    else:
        print("\n❌ Configuration issues detected. Please update your configuration file.")
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
