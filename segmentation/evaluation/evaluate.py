import os
import argparse
import pandas as pd
import numpy as np
import tifffile
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import importlib
from typing import Dict, Any, List
from tqdm import tqdm

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config

def load_3d_tif(file_path: str) -> np.ndarray:
    """
    Loads a 3D TIFF file from the given path.
    """
    return tifffile.imread(file_path)

def get_files_dict(folder: str, extension: str = ".tif") -> Dict[str, str]:
    """
    Returns a dictionary mapping the base filename (without extension)
    to its full file path for all files with the given extension in the folder.
    """
    file_dict: Dict[str, str] = {}
    for file in os.listdir(folder):
        if file.lower().endswith(extension):
            base, _ = os.path.splitext(file)
            file_dict[base] = os.path.join(folder, file)
    return file_dict

def compute_metrics(label: np.ndarray, prediction: np.ndarray, metrics_config: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes metrics based on the provided configuration.
    
    For each metric specified in metrics_config, this function dynamically
    imports a module from the 'metrics' folder and calls its 'compute' function.
    
    Args:
        label (np.ndarray): Ground truth 3D array.
        prediction (np.ndarray): Predicted 3D array.
        metrics_config (List[Dict[str, Any]]): List of metric configuration dictionaries.
        
    Returns:
        Dict[str, float]: Dictionary of computed metric values.
    """
    results: Dict[str, float] = {}
    for metric in metrics_config:
        metric_type: str = metric["type"]
        hyperparams: Dict[str, Any] = metric.get("hyperparameters", {})
        try:
            mod = importlib.import_module(f"metrics.{metric_type}")
            metric_value = mod.compute(label, prediction, **hyperparams)
            results[metric_type] = float(metric_value)
        except ModuleNotFoundError:
            print(f"Module for metric '{metric_type}' not found in the metrics folder. Skipping.")
        except AttributeError:
            print(f"Module for metric '{metric_type}' does not have a 'compute' function. Skipping.")
    return results

def log_histogram_for_metric(
    df: pd.DataFrame,
    metric: str,
    output_folder: str,
    bins: int = 20,
    color: str = "skyblue"
) -> None:
    """
    Creates a histogram for the specified metric using Seaborn and logs it to wandb.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metric data.
        metric (str): Column name of the metric to plot.
        output_folder (str): Directory where the histogram image will be saved.
        bins (int, optional): Number of bins in the histogram. Defaults to 20.
        color (str, optional): Color palette for the histogram. Defaults to "skyblue".
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in the results. Skipping histogram plot.")
        return

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(df[metric], bins=bins, kde=True, color=color)
    ax.set_title(f"Distribution of {metric}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    plt.tight_layout()

    plot_path = os.path.join(output_folder, f"{metric}_hist.png")
    plt.savefig(plot_path)
    plt.close()
    wandb.log({f"{metric}_hist": wandb.Image(plot_path)})
    print(f"Histogram for metric '{metric}' saved to {plot_path}")

def compute_statistics(df: pd.DataFrame, metric: str) -> Dict[str, float]:
    """
    Computes robust statistical summaries for the given metric.
    
    Returns a dictionary with keys grouped as '{metric}/{statistic}'.
    """
    values = df[metric]
    stats = {
        f"{metric}/avg": values.mean(),
        f"{metric}/median": values.median(),
        f"{metric}/std": values.std(),
        f"{metric}/q1": values.quantile(0.25),
        f"{metric}/q3": values.quantile(0.75),
        f"{metric}/min": values.min(),
        f"{metric}/max": values.max(),
    }
    return stats

def main(args: argparse.Namespace) -> None:
    # Load configuration from YAML
    config: Dict[str, Any] = load_config(args.config)
    labels_folder: str = config["labels_folder"]
    predictions_folder: str = config["predictions_folder"]
    output_csv: str = config["output_csv"]
    output_folder: str = config["output_folder"]
    wandb_project: str = config["wandb_project"]
    run_name: str = config["run_name"]
    metrics_config: List[Dict[str, Any]] = config.get("metrics", [])
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize Weights & Biases
    wandb.init(project=wandb_project, name=run_name, config=config)
    
    # Build dictionaries mapping file base names to file paths
    label_files: Dict[str, str] = get_files_dict(labels_folder)
    pred_files: Dict[str, str] = get_files_dict(predictions_folder)
    
    # Find common files based on their base names
    common_bases = set(label_files.keys()).intersection(set(pred_files.keys()))
    if not common_bases:
        print("No matching files found between the two folders.")
        return
    
    results: List[Dict[str, Any]] = []
    
    # Process each file pair with a progress bar
    for base in tqdm(common_bases, desc="Processing files"):
        label_path: str = label_files[base]
        pred_path: str = pred_files[base]
        
        label_data: np.ndarray = load_3d_tif(label_path)
        pred_data: np.ndarray = load_3d_tif(pred_path)
        
        # Compute metrics as per configuration
        metrics: Dict[str, float] = compute_metrics(label_data, pred_data, metrics_config)
        # Keep the file name for CSV output only
        metrics_with_file = metrics.copy()
        metrics_with_file["file"] = base
        results.append(metrics_with_file)
        
        # Log only numeric metrics to wandb
        wandb.log(metrics)
    
    # Save the detailed results to CSV (including file names)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Metrics saved to {output_csv}")
    
    # Compute and log robust statistical summaries for each metric defined in the configuration
    stats_summary: Dict[str, float] = {}
    metric_names = [m["type"] for m in metrics_config]
    for metric in metric_names:
        if metric in df.columns:
            stats_summary.update(compute_statistics(df, metric))
    wandb.log(stats_summary)
    
    # Log histograms for each metric defined in the configuration using Seaborn
    for metric in metric_names:
        log_histogram_for_metric(df, metric=metric, output_folder=output_folder)
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for 3D .tif files using YAML configuration, dynamic metric import, tqdm, and robust statistical logging."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args)
