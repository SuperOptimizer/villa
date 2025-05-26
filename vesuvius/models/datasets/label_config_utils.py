"""
Utility functions and examples for multi-task and multi-class label handling.

This module provides helper functions to configure the vesuvius system for 
different labeling scenarios, including multi-task learning and multi-class 
segmentation tasks.
"""

import numpy as np
from typing import Dict, Union, Any, List, Tuple, Optional


def create_binary_task_config(targets: Union[str, List[str]], foreground_value: int = 1) -> Dict[str, Any]:
    """
    Create a configuration for binary segmentation task(s).
    
    Parameters
    ----------
    targets : str or list of str
        Name(s) of the target task(s) (e.g., "ink" or ["ink", "damage"])
    foreground_value : int, default=1
        Value to assign to foreground pixels
        
    Returns
    -------
    dict
        Configuration dictionary for the task(s)
    """
    if isinstance(targets, str):
        targets = [targets]
    
    target_values = {target: foreground_value for target in targets}
    
    return {
        "binarize_labels": True,
        "target_value": target_values
    }


def create_multitask_config(task_configs: Dict[str, Union[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Create a configuration for multi-task learning with different task types.
    
    Parameters
    ----------
    task_configs : dict
        Dictionary mapping task names to their configurations.
        - For binary tasks: {"task_name": foreground_value}
        - For multi-class tasks: {"task_name": {"type": "multiclass", "mapping": {0: 0, 1: 1, ...}}}
        
    Returns
    -------
    dict
        Configuration dictionary for multi-task learning
    """
    target_values = {}
    
    for task_name, config in task_configs.items():
        if isinstance(config, int):
            # Simple binary task
            target_values[task_name] = config
        elif isinstance(config, dict) and config.get("type") == "multiclass":
            # Multi-class task with mapping
            target_values[task_name] = config["mapping"]
        else:
            raise ValueError(f"Invalid configuration for task '{task_name}'")
    
    return {
        "binarize_labels": True,
        "target_value": target_values
    }


def create_multiclass_config(target_name: str, class_mapping: Optional[Dict[int, int]] = None, 
                            num_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a configuration for multi-class segmentation with optional label remapping.
    
    Parameters
    ----------
    target_name : str
        Name of the target task
    class_mapping : dict, optional
        Dictionary mapping original class labels to new class labels
        Example: {0: 0, 1: 1, 2: 2, 3: 2, 4: 3} (merges classes 3 into 2)
    num_classes : int, optional
        If provided without class_mapping, creates identity mapping for classes 0 to num_classes-1
        
    Returns
    -------
    dict
        Configuration dictionary for multi-class segmentation
    """
    if class_mapping is None and num_classes is None:
        raise ValueError("Either class_mapping or num_classes must be provided")
    
    if class_mapping is None:
        # Create identity mapping
        class_mapping = {i: i for i in range(num_classes)}
    
    return {
        "binarize_labels": True,
        "target_value": {target_name: class_mapping}
    }


def create_multiclass_with_regions_config(target_name: str, 
                                        class_mapping: Optional[Dict[int, int]] = None,
                                        regions: Optional[Dict[int, List[int]]] = None,
                                        num_classes: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a configuration for multi-class segmentation with region combinations.
    
    Regions allow you to define new classes as combinations of existing classes.
    When regions are applied, pixels belonging to any of the specified classes
    will be overridden to become the region class.
    
    Parameters
    ----------
    target_name : str
        Name of the target task
    class_mapping : dict, optional
        Dictionary mapping original class labels to new class labels
        Example: {0: 0, 1: 1, 2: 2, 3: 3}
    regions : dict, optional
        Dictionary mapping new region IDs to lists of classes to combine
        Example: {4: [1, 2], 5: [2, 3]} creates region 4 from classes 1&2
    num_classes : int, optional
        If provided without class_mapping, creates identity mapping
        
    Returns
    -------
    dict
        Configuration dictionary for multi-class segmentation with regions
    """
    if class_mapping is None and num_classes is None:
        raise ValueError("Either class_mapping or num_classes must be provided")
    
    if class_mapping is None:
        # Create identity mapping
        class_mapping = {i: i for i in range(num_classes)}
    
    config_value = {"mapping": class_mapping}
    
    if regions:
        # Validate regions don't conflict with mapped classes
        mapped_values = set(class_mapping.values())
        for region_id, source_classes in regions.items():
            if region_id in mapped_values:
                raise ValueError(
                    f"Region ID {region_id} conflicts with existing mapped class. "
                    f"Mapped classes: {sorted(mapped_values)}"
                )
            if not isinstance(source_classes, list):
                raise ValueError(
                    f"Region {region_id} must specify a list of source classes, "
                    f"got {type(source_classes).__name__}"
                )
        config_value["regions"] = regions
    
    return {
        "binarize_labels": True,
        "target_value": {target_name: config_value}
    }


def create_no_binarization_config() -> Dict[str, Any]:
    """
    Create a configuration for when labels are already properly formatted.
    
    Returns
    -------
    dict
        Configuration dictionary that skips binarization
    """
    return {
        "binarize_labels": False,
        "target_value": None  # Not used when binarize_labels is False
    }


def validate_label_data(labels: np.ndarray, target_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Validate label data and check against configuration.
    
    Parameters
    ----------
    labels : np.ndarray
        Label array to analyze
    target_config : dict, optional
        Target configuration to validate against
        
    Returns
    -------
    dict
        Analysis results and validation status
    """
    unique_values = np.unique(labels)
    unique_values = unique_values[unique_values >= 0]  # Exclude negative values if any
    
    analysis = {
        "unique_values": unique_values.tolist(),
        "num_classes": len(unique_values),
        "has_background": 0 in unique_values,
        "value_range": (int(unique_values.min()), int(unique_values.max())),
        "class_counts": {},
        "validation": {"status": "passed", "issues": []},
        "recommendations": []
    }
    
    # Count pixels per class
    for val in unique_values:
        count = np.sum(labels == val)
        analysis["class_counts"][int(val)] = int(count)
    
    # If target configuration provided, validate against it
    if target_config:
        if isinstance(target_config, int):
            # Binary task configuration
            expected_values = [0, target_config]
            unexpected = set(unique_values) - set(expected_values)
            if unexpected:
                analysis["validation"]["status"] = "failed"
                analysis["validation"]["issues"].append(
                    f"Unexpected label values: {sorted(unexpected)}. Expected only 0 and {target_config}"
                )
        elif isinstance(target_config, dict):
            # Multi-class configuration
            configured_values = set(target_config.keys())
            label_values = set(int(v) for v in unique_values if v > 0)  # Exclude background
            
            missing_in_config = label_values - configured_values
            if missing_in_config:
                analysis["validation"]["status"] = "warning"
                analysis["validation"]["issues"].append(
                    f"Label values {sorted(missing_in_config)} found in data but not in configuration"
                )
                analysis["recommendations"].append(
                    f"Add mapping for values {sorted(missing_in_config)} to your configuration"
                )
    
    # General recommendations
    if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
        analysis["recommendations"].append("Binary labels detected (0, 1). Configuration looks good.")
    elif len(unique_values) == 2 and 0 in unique_values:
        foreground_val = [v for v in unique_values if v != 0][0]
        analysis["recommendations"].append(
            f"Binary task with foreground value {foreground_val}. Ensure target_value is set to {foreground_val}"
        )
    elif len(unique_values) > 2:
        analysis["recommendations"].append(
            "Multi-class data detected. Ensure class_mapping is properly configured for all classes."
        )
    
    return analysis


def analyze_dataset_labels(dataset, sample_size: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Analyze labels from a dataset by sampling patches.
    
    Parameters
    ----------
    dataset : BaseDataset
        The dataset to analyze
    sample_size : int, default=10
        Number of patches to sample for analysis
        
    Returns
    -------
    dict
        Analysis results for each target
    """
    results = {}
    
    # Sample patches
    indices = np.random.choice(len(dataset), size=min(sample_size, len(dataset)), replace=False)
    
    for idx in indices:
        data = dataset[idx]
        
        for target_name in dataset.targets.keys():
            if target_name not in results:
                results[target_name] = {
                    "all_unique_values": set(),
                    "samples": []
                }
            
            if target_name in data:
                label_data = data[target_name].numpy()
                unique_vals = np.unique(label_data)
                results[target_name]["all_unique_values"].update(int(v) for v in unique_vals)
                results[target_name]["samples"].append({
                    "patch_idx": int(idx),
                    "unique_values": unique_vals.tolist(),
                    "shape": label_data.shape
                })
    
    # Analyze collected data
    for target_name, target_data in results.items():
        target_data["all_unique_values"] = sorted(list(target_data["all_unique_values"]))
        target_data["num_unique_values"] = len(target_data["all_unique_values"])
        
        # Get target configuration
        target_config = None
        if hasattr(dataset, 'target_value') and isinstance(dataset.target_value, dict):
            target_config = dataset.target_value.get(target_name)
        
        # Validate if configuration exists
        if target_config:
            if isinstance(target_config, dict):
                configured_values = set(target_config.keys())
                found_values = set(target_data["all_unique_values"]) - {0}  # Exclude background
                missing_in_config = found_values - configured_values
                if missing_in_config:
                    target_data["validation_warning"] = (
                        f"Values {sorted(missing_in_config)} found but not in configuration"
                    )
    
    return results


def print_scenario_examples():
    """
    Print example configurations for different scenarios.
    """
    print("=== Multi-task and Multi-class Configuration Examples ===\n")
    
    print("SCENARIO 1: Simple Binary Task")
    print("Use case: Single binary segmentation task")
    config1 = create_binary_task_config("ink", foreground_value=1)
    print(f"Configuration: {config1}")
    print("In YAML:")
    print("""  binarize_labels: true
  target_value:
    ink: 1
""")
    
    print("\nSCENARIO 2: Multi-task Binary Segmentation")
    print("Use case: Multiple binary tasks from same input image")
    config2 = create_binary_task_config(["ink", "damage", "texture"], foreground_value=1)
    print(f"Configuration: {config2}")
    print("In YAML:")
    print("""  binarize_labels: true
  target_value:
    ink: 1
    damage: 1
    texture: 1
""")
    
    print("\nSCENARIO 3: Multi-class Segmentation")
    print("Use case: Single task with multiple classes")
    config3 = create_multiclass_config("segmentation", {0: 0, 1: 1, 2: 2, 3: 2, 4: 3})
    print(f"Configuration: {config3}")
    print("This maps: background(0)->0, class1(1)->1, class2(2)->2, class3(3)->2 (merged), class4(4)->3")
    print("In YAML:")
    print("""  binarize_labels: true
  target_value:
    segmentation:
      0: 0  # background
      1: 1  # class 1
      2: 2  # class 2
      3: 2  # merge class 3 into 2
      4: 3  # class 4 becomes 3
""")
    
    print("\nSCENARIO 4: Mixed Multi-task (Binary + Multi-class)")
    print("Use case: Combination of binary and multi-class tasks")
    config4 = create_multitask_config({
        "ink": 1,  # Binary task
        "damage": 1,  # Binary task
        "material_type": {"type": "multiclass", "mapping": {0: 0, 1: 1, 2: 2, 3: 3}}  # Multi-class
    })
    print(f"Configuration: {config4}")
    print("In YAML:")
    print("""  binarize_labels: true
  target_value:
    ink: 1
    damage: 1
    material_type:
      0: 0  # background
      1: 1  # material type 1
      2: 2  # material type 2
      3: 3  # material type 3
""")
    
    print("\nSCENARIO 5: Multi-class with Regions")
    print("Use case: Multi-class segmentation with region combinations")
    config5 = create_multiclass_with_regions_config(
        "tissue_segmentation",
        class_mapping={0: 0, 1: 1, 2: 2, 3: 3},
        regions={4: [1, 2], 5: [2, 3], 6: [1, 2, 3]}
    )
    print(f"Configuration: {config5}")
    print("This creates regions that combine existing classes:")
    print("  - Region 4 = pixels from class 1 OR 2 (e.g., all living tissue)")
    print("  - Region 5 = pixels from class 2 OR 3 (e.g., all abnormal tissue)")
    print("  - Region 6 = pixels from class 1, 2, OR 3 (e.g., all tissue types)")
    print("In YAML:")
    print("""  binarize_labels: true
  target_value:
    tissue_segmentation:
      mapping:
        0: 0  # background
        1: 1  # healthy tissue
        2: 2  # damaged tissue
        3: 3  # scar tissue
      regions:
        4: [1, 2]      # all living tissue
        5: [2, 3]      # all abnormal tissue
        6: [1, 2, 3]   # all tissue types
""")
    
    print("\nSCENARIO 6: Pre-formatted Labels")
    print("Use case: Labels are already in the correct format (0/1 for binary, 0/1/2/... for multi-class)")
    config6 = create_no_binarization_config()
    print(f"Configuration: {config6}")
    print("In YAML:")
    print("""  binarize_labels: false
  target_value: null  # Not used when binarize_labels is false
""")


if __name__ == "__main__":
    print_scenario_examples()
