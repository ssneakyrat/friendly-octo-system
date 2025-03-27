#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml
import time
import torch
import random
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np

from data.datasets import FutureVoxDataset


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config)
    
    return config


def benchmark_dataset(dataset, num_samples=100, batch_size=16, num_workers=4, use_tqdm=True):
    """
    Benchmark dataset loading speed
    
    Args:
        dataset: Dataset to benchmark
        num_samples: Number of samples to load
        batch_size: Batch size
        num_workers: Number of workers
        use_tqdm: Whether to use tqdm for progress display
        
    Returns:
        Dictionary of benchmark results
    """
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Randomly sample indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Measure individual item loading time
    item_times = []
    if use_tqdm:
        iterator = tqdm(indices, desc="Benchmarking individual items")
    else:
        iterator = indices
    
    for idx in iterator:
        start_time = time.time()
        _ = dataset[idx]
        item_times.append(time.time() - start_time)
    
    # Measure batch loading time
    batch_times = []
    num_batches = min(num_samples // batch_size, len(dataloader))
    
    if use_tqdm:
        iterator = tqdm(enumerate(dataloader), desc="Benchmarking batches", total=num_batches)
    else:
        iterator = enumerate(dataloader)
    
    for i, batch in iterator:
        if i >= num_batches:
            break
        batch_times.append(dataloader.sampler.sample_rate if hasattr(dataloader.sampler, 'sample_rate') else 0)
    
    # Calculate statistics
    avg_item_time = sum(item_times) / len(item_times)
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    
    # Calculate throughput
    items_per_second = 1.0 / avg_item_time if avg_item_time > 0 else 0
    batches_per_second = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
    samples_per_second = batches_per_second * batch_size
    
    return {
        'item_times': item_times,
        'batch_times': batch_times,
        'avg_item_time': avg_item_time,
        'avg_batch_time': avg_batch_time,
        'items_per_second': items_per_second,
        'batches_per_second': batches_per_second,
        'samples_per_second': samples_per_second
    }


def compare_dataset_formats(config, num_samples=100, batch_size=16, num_workers=4):
    """
    Compare loading speed of standard vs binary dataset formats
    
    Args:
        config: Configuration object
        num_samples: Number of samples to load
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    # Load phoneme dictionary
    phoneme_dict = {}
    if Path(config.preprocessing.text.phoneme_dict).exists():
        import json
        with open(config.preprocessing.text.phoneme_dict, 'r') as f:
            phoneme_dict = json.load(f)
    
    # Check if binary files are available
    binary_available = (
        hasattr(config, 'binary') and
        hasattr(config.binary, 'train_file') and
        Path(config.binary.train_file).exists()
    )
    
    # Create standard dataset
    print("Creating standard dataset...")
    standard_dataset = FutureVoxDataset(
        data_dir=Path(config.data_dir) / "train",
        phoneme_dict=phoneme_dict,
        sample_rate=config.preprocessing.audio.sample_rate,
        n_fft=config.preprocessing.audio.n_fft,
        hop_length=config.preprocessing.audio.hop_length,
        win_length=config.preprocessing.audio.win_length,
        n_mels=config.preprocessing.audio.n_mels,
        f_min=config.preprocessing.audio.f_min,
        f_max=config.preprocessing.audio.f_max,
        f0_min=config.preprocessing.f0.min_f0,
        f0_max=config.preprocessing.f0.max_f0,
        g2p_model=config.preprocessing.text.g2p_model,
        cleaner=config.preprocessing.text.cleaner,
        binary_file=None
    )
    
    # Benchmark standard dataset
    print(f"Benchmarking standard dataset with {len(standard_dataset)} items...")
    standard_results = benchmark_dataset(
        standard_dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers
    )
    results['standard'] = standard_results
    
    # Create and benchmark binary dataset if available
    if binary_available:
        print("Creating binary dataset...")
        binary_dataset = FutureVoxDataset(
            data_dir=Path(config.data_dir) / "train",
            phoneme_dict=phoneme_dict,
            sample_rate=config.preprocessing.audio.sample_rate,
            n_fft=config.preprocessing.audio.n_fft,
            hop_length=config.preprocessing.audio.hop_length,
            win_length=config.preprocessing.audio.win_length,
            n_mels=config.preprocessing.audio.n_mels,
            f_min=config.preprocessing.audio.f_min,
            f_max=config.preprocessing.audio.f_max,
            f0_min=config.preprocessing.f0.min_f0,
            f0_max=config.preprocessing.f0.max_f0,
            g2p_model=config.preprocessing.text.g2p_model,
            cleaner=config.preprocessing.text.cleaner,
            binary_file=config.binary.train_file
        )
        
        print(f"Benchmarking binary dataset with {len(binary_dataset)} items...")
        binary_results = benchmark_dataset(
            binary_dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            num_workers=num_workers
        )
        results['binary'] = binary_results
    else:
        print("Binary dataset not available, skipping binary benchmark")
    
    return results


def plot_benchmark_results(results, output_path=None):
    """
    Plot benchmark results
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to save plot
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot item loading time distribution
    axs[0, 0].set_title('Item Loading Time Distribution')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Frequency')
    
    for format_name, format_results in results.items():
        axs[0, 0].hist(format_results['item_times'], alpha=0.5, label=format_name)
    
    axs[0, 0].legend()
    
    # Plot batch loading time distribution
    axs[0, 1].set_title('Batch Loading Time Distribution')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Frequency')
    
    for format_name, format_results in results.items():
        axs[0, 1].hist(format_results['batch_times'], alpha=0.5, label=format_name)
    
    axs[0, 1].legend()
    
    # Plot average loading times
    formats = list(results.keys())
    avg_item_times = [results[fmt]['avg_item_time'] for fmt in formats]
    avg_batch_times = [results[fmt]['avg_batch_time'] for fmt in formats]
    
    x = np.arange(len(formats))
    width = 0.35
    
    axs[1, 0].bar(x - width/2, avg_item_times, width, label='Item Time')
    axs[1, 0].bar(x + width/2, avg_batch_times, width, label='Batch Time')
    
    axs[1, 0].set_title('Average Loading Times')
    axs[1, 0].set_xlabel('Format')
    axs[1, 0].set_ylabel('Time (s)')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(formats)
    axs[1, 0].legend()
    
    # Plot throughput
    items_per_second = [results[fmt]['items_per_second'] for fmt in formats]
    samples_per_second = [results[fmt]['samples_per_second'] for fmt in formats]
    
    axs[1, 1].bar(x - width/2, items_per_second, width, label='Items/second')
    axs[1, 1].bar(x + width/2, samples_per_second, width, label='Samples/second')
    
    axs[1, 1].set_title('Throughput')
    axs[1, 1].set_xlabel('Format')
    axs[1, 1].set_ylabel('Items/Samples per second')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(formats)
    axs[1, 1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path)
        print(f"Benchmark plot saved to {output_path}")
    
    # Show figure
    plt.show()


def print_benchmark_results(results):
    """
    Print benchmark results
    
    Args:
        results: Benchmark results dictionary
    """
    print("\n=== Benchmark Results ===")
    
    for format_name, format_results in results.items():
        print(f"\n{format_name.capitalize()} Format:")
        print(f"  • Average item loading time: {format_results['avg_item_time']:.6f} seconds")
        print(f"  • Average batch loading time: {format_results['avg_batch_time']:.6f} seconds")
        print(f"  • Items per second: {format_results['items_per_second']:.2f}")
        print(f"  • Batches per second: {format_results['batches_per_second']:.2f}")
        print(f"  • Samples per second: {format_results['samples_per_second']:.2f}")
    
    # Print comparison if both formats are available
    if 'standard' in results and 'binary' in results:
        std_item_time = results['standard']['avg_item_time']
        bin_item_time = results['binary']['avg_item_time']
        
        std_batch_time = results['standard']['avg_batch_time']
        bin_batch_time = results['binary']['avg_batch_time']
        
        item_speedup = std_item_time / bin_item_time if bin_item_time > 0 else float('inf')
        batch_speedup = std_batch_time / bin_batch_time if bin_batch_time > 0 else float('inf')
        
        print("\nPerformance Comparison:")
        print(f"  • Item loading speedup with binary: {item_speedup:.2f}x")
        print(f"  • Batch loading speedup with binary: {batch_speedup:.2f}x")


def main():
    """
    Benchmark FutureVox+ dataset loading
    """
    parser = argparse.ArgumentParser(description="Benchmark FutureVox+ dataset loading")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to load")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--output_plot", type=str, help="Path to save benchmark plot")
    parser.add_argument("--skip_plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load configuration
    config = load_config(args.config)
    
    # Run benchmark
    results = compare_dataset_formats(
        config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Print results
    print_benchmark_results(results)
    
    # Plot results
    if not args.skip_plot:
        plot_benchmark_results(results, args.output_plot)


if __name__ == "__main__":
    main()