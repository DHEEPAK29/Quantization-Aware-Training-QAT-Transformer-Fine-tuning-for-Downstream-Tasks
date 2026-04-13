"""
Comprehensive visualization of weight clustering during QAT training.
Generates histograms, statistics, and clustering analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import seaborn as sns

sns.set_style("whitegrid")


class VisualizationConfig:
    """Configuration for visualization."""
    weight_log_dir = Path("./qat_outputs/weight_logs")
    output_dir = Path("./qat_outputs/visualizations")
    metrics_file = Path("./qat_outputs/metrics.json")
    
    # Histogram settings
    n_bins = 200
    xlim_auto = True  # Auto-detect range instead of hardcoding
    xlim_manual = (-0.1, 0.1)  # Fallback if auto fails


def load_weight_files(weight_log_dir):
    """
    Load all weight log files organized by layer and epoch.
    
    Args:
        weight_log_dir: Directory containing weight logs
        
    Returns:
        dict: Organized weight tensors
    """
    weight_log_dir = Path(weight_log_dir)
    
    if not weight_log_dir.exists():
        raise FileNotFoundError(f"Weight log directory not found: {weight_log_dir}")
    
    # Organize by layer type
    weights_by_layer = defaultdict(list)
    
    for weight_file in sorted(weight_log_dir.glob("*.pt")):
        # Parse filename: layer_type_epoch_step.pt
        parts = weight_file.stem.split("_")
        
        # Extract layer type (e.g., "mlp_fc1" or "attn_q")
        if "mlp" in weight_file.name:
            layer_type = "mlp_fc1"
        elif "attn" in weight_file.name:
            layer_type = "attn_q"
        else:
            layer_type = "unknown"
        
        weights = torch.load(weight_file, map_location='cpu').numpy()
        
        weights_by_layer[layer_type].append({
            'file': weight_file.name,
            'weights': weights,
        })
    
    return dict(weights_by_layer)


def compute_clustering_metrics(weights, n_bins=50):
    """
    Compute metrics that indicate weight clustering.
    
    Args:
        weights: Numpy array of weights
        n_bins: Number of bins for histogram analysis
        
    Returns:
        dict: Clustering metrics
    """
    # Basic statistics
    metrics = {
        'mean': float(np.mean(weights)),
        'std': float(np.std(weights)),
        'min': float(np.min(weights)),
        'max': float(np.max(weights)),
        'unique_values': len(np.unique(weights)),
        'total_values': len(weights),
    }
    
    # Histogram analysis
    hist, bin_edges = np.histogram(weights, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Entropy (lower entropy = more clustering)
    hist_normalized = hist / hist.sum()
    hist_normalized = hist_normalized[hist_normalized > 0]  # Remove zeros
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
    metrics['entropy'] = float(entropy)
    
    # Peak concentration (what % of weights are in top 10% of bins)
    top_bins_idx = np.argsort(hist)[-max(1, n_bins // 10):]
    peak_concentration = hist[top_bins_idx].sum() / hist.sum()
    metrics['peak_concentration'] = float(peak_concentration)
    
    # Effective number of bins (bins with >1% of weights)
    effective_bins = np.sum(hist > 0.01 * hist.sum())
    metrics['effective_bins'] = int(effective_bins)
    
    # Quantization grid alignment (for INT4/INT8)
    # Count how many weights are close to integer grid values
    scale = np.max(np.abs(weights))
    if scale > 0:
        # Simulate INT8 grid (-127 to 127)
        quantized = np.round(weights / scale * 127) / 127 * scale
        quantization_error = np.mean(np.abs(weights - quantized))
        metrics['int8_quantization_error'] = float(quantization_error)
    
    return metrics


def plot_weight_evolution(weights_by_layer, config: VisualizationConfig):
    """
    Create histogram evolution plots for each layer.
    
    Args:
        weights_by_layer: Dictionary of weight tensors by layer
        config: VisualizationConfig instance
    """
    config.output_dir.mkdir(exist_ok=True, parents=True)
    
    for layer_name, weight_history in weights_by_layer.items():
        n_snapshots = len(weight_history)
        
        if n_snapshots == 0:
            continue
        
        # Create subplot grid
        n_cols = min(4, n_snapshots)
        n_rows = (n_snapshots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        if n_snapshots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Determine global xlim if auto mode
        if config.xlim_auto:
            all_weights = np.concatenate([w['weights'] for w in weight_history])
            percentile_low = np.percentile(all_weights, 1)
            percentile_high = np.percentile(all_weights, 99)
            xlim = (percentile_low, percentile_high)
        else:
            xlim = config.xlim_manual
        
        # Plot each snapshot
        for i, weight_data in enumerate(weight_history):
            weights = weight_data['weights']
            file_name = weight_data['file']
            
            # Compute metrics
            metrics = compute_clustering_metrics(weights, n_bins=config.n_bins)
            
            # Plot histogram
            axes[i].hist(
                weights,
                bins=config.n_bins,
                color='royalblue',
                alpha=0.7,
                edgecolor='black',
                linewidth=0.5
            )
            
            axes[i].set_xlim(xlim)
            axes[i].set_title(
                f"{file_name}\n"
                f"Entropy: {metrics['entropy']:.2f}, "
                f"Unique: {metrics['unique_values']}"
            )
            axes[i].set_xlabel("Weight Value")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(
            config.output_dir / f"{layer_name}_evolution.png",
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"✓ Saved {layer_name} evolution plot")


def plot_clustering_metrics_over_time(weights_by_layer, config: VisualizationConfig):
    """
    Plot how clustering metrics evolve during training.
    
    Args:
        weights_by_layer: Dictionary of weight tensors by layer
        config: VisualizationConfig instance
    """
    for layer_name, weight_history in weights_by_layer.items():
        if len(weight_history) < 2:
            continue
        
        # Compute metrics for each snapshot
        metrics_history = []
        for weight_data in weight_history:
            metrics = compute_clustering_metrics(weight_data['weights'])
            metrics_history.append(metrics)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        steps = range(len(metrics_history))
        
        # Entropy
        axes[0, 0].plot(
            steps,
            [m['entropy'] for m in metrics_history],
            marker='o',
            linewidth=2
        )
        axes[0, 0].set_title("Entropy (Lower = More Clustering)")
        axes[0, 0].set_xlabel("Snapshot")
        axes[0, 0].set_ylabel("Entropy")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Peak concentration
        axes[0, 1].plot(
            steps,
            [m['peak_concentration'] for m in metrics_history],
            marker='o',
            color='green',
            linewidth=2
        )
        axes[0, 1].set_title("Peak Concentration (Higher = More Clustering)")
        axes[0, 1].set_xlabel("Snapshot")
        axes[0, 1].set_ylabel("Concentration")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Effective bins
        axes[1, 0].plot(
            steps,
            [m['effective_bins'] for m in metrics_history],
            marker='o',
            color='orange',
            linewidth=2
        )
        axes[1, 0].set_title("Effective Bins (Lower = More Clustering)")
        axes[1, 0].set_xlabel("Snapshot")
        axes[1, 0].set_ylabel("Number of Bins")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Quantization error
        if 'int8_quantization_error' in metrics_history[0]:
            axes[1, 1].plot(
                steps,
                [m['int8_quantization_error'] for m in metrics_history],
                marker='o',
                color='red',
                linewidth=2
            )
            axes[1, 1].set_title("INT8 Quantization Error (Lower = Better Alignment)")
            axes[1, 1].set_xlabel("Snapshot")
            axes[1, 1].set_ylabel("Error")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.suptitle(f"{layer_name} - Clustering Metrics Over Training", fontsize=14)
        plt.tight_layout()
        plt.savefig(
            config.output_dir / f"{layer_name}_metrics.png",
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()
        
        print(f"✓ Saved {layer_name} metrics plot")


def plot_training_loss(config: VisualizationConfig):
    """
    Plot training loss curve if metrics file exists.
    
    Args:
        config: VisualizationConfig instance
    """
    if not config.metrics_file.exists():
        print(f"⚠ Metrics file not found: {config.metrics_file}")
        return
    
    with open(config.metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if 'losses' not in metrics or len(metrics['losses']) == 0:
        print("⚠ No loss data found in metrics")
        return
    
    # Extract loss values
    steps = [entry['step'] for entry in metrics['losses']]
    losses = [entry['loss'] for entry in metrics['losses']]
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        config.output_dir / "training_loss.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()
    
    print("✓ Saved training loss plot")


def generate_report(weights_by_layer, config: VisualizationConfig):
    """
    Generate a text report summarizing clustering results.
    
    Args:
        weights_by_layer: Dictionary of weight tensors by layer
        config: VisualizationConfig instance
    """
    report_lines = [
        "="*80,
        "QAT WEIGHT CLUSTERING REPORT",
        "="*80,
        ""
    ]
    
    for layer_name, weight_history in weights_by_layer.items():
        if len(weight_history) == 0:
            continue
        
        report_lines.append(f"\n{layer_name.upper()}")
        report_lines.append("-" * 40)
        
        # Initial state
        initial_metrics = compute_clustering_metrics(weight_history[0]['weights'])
        report_lines.append(f"Initial state ({weight_history[0]['file']}):")
        report_lines.append(f"  Entropy: {initial_metrics['entropy']:.4f}")
        report_lines.append(f"  Unique values: {initial_metrics['unique_values']:,}")
        report_lines.append(f"  Peak concentration: {initial_metrics['peak_concentration']:.2%}")
        
        # Final state
        if len(weight_history) > 1:
            final_metrics = compute_clustering_metrics(weight_history[-1]['weights'])
            report_lines.append(f"\nFinal state ({weight_history[-1]['file']}):")
            report_lines.append(f"  Entropy: {final_metrics['entropy']:.4f}")
            report_lines.append(f"  Unique values: {final_metrics['unique_values']:,}")
            report_lines.append(f"  Peak concentration: {final_metrics['peak_concentration']:.2%}")
            
            # Changes
            entropy_change = ((final_metrics['entropy'] - initial_metrics['entropy']) 
                             / initial_metrics['entropy'] * 100)
            report_lines.append(f"\nChanges:")
            report_lines.append(f"  Entropy change: {entropy_change:+.1f}%")
            report_lines.append(f"  Peak concentration change: "
                              f"{(final_metrics['peak_concentration'] - initial_metrics['peak_concentration']):.2%}")
    
    report_lines.append("\n" + "="*80)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open(config.output_dir / "clustering_report.txt", "w") as f:
        f.write(report_text)
    
    # Print to console
    print("\n" + report_text)


def main():
    """Run all visualization tasks."""
    config = VisualizationConfig()
    
    print("Loading weight logs...")
    weights_by_layer = load_weight_files(config.weight_log_dir)
    
    if not weights_by_layer:
        print(f"⚠ No weight logs found in {config.weight_log_dir}")
        print("Make sure you've run train.py first!")
        return
    
    print(f"Found weight logs for {len(weights_by_layer)} layers")
    
    print("\nGenerating visualizations...")
    plot_weight_evolution(weights_by_layer, config)
    plot_clustering_metrics_over_time(weights_by_layer, config)
    plot_training_loss(config)
    
    print("\nGenerating report...")
    generate_report(weights_by_layer, config)
    
    print(f"\n✓ All visualizations saved to {config.output_dir}")


if __name__ == "__main__":
    main()