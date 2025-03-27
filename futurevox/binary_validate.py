#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml
import json
from pathlib import Path
from omegaconf import OmegaConf

from preprocessing.binary import validate_binary_file


def main():
    """
    Validate FutureVox+ binary dataset files
    """
    parser = argparse.ArgumentParser(description="Validate FutureVox+ binary dataset files")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--binary_file", type=str, help="Path to binary file to validate (overrides config)")
    parser.add_argument("--metadata_file", type=str, help="Path to metadata file for comparison (optional)")
    parser.add_argument("--validate_all", action="store_true", help="Validate all binary files in configuration")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)
    
    # If validate_all is specified, validate all binary files in config
    if args.validate_all:
        if hasattr(config, 'binary'):
            binary_files = []
            if hasattr(config.binary, 'train_file'):
                binary_files.append(('train', config.binary.train_file))
            if hasattr(config.binary, 'val_file'):
                binary_files.append(('validation', config.binary.val_file))
            if hasattr(config.binary, 'test_file'):
                binary_files.append(('test', config.binary.test_file))
            
            if not binary_files:
                print("No binary files specified in configuration")
                return
            
            # Validate each file
            all_valid = True
            for file_type, binary_file in binary_files:
                binary_path = Path(binary_file)
                if binary_path.exists():
                    print(f"\nValidating {file_type} binary file: {binary_path}")
                    validation_results = validate_binary_file(binary_path)
                    all_valid = all_valid and validation_results['valid']
                    print_validation_results(validation_results)
                else:
                    print(f"\n{file_type.capitalize()} binary file not found: {binary_path}")
                    all_valid = False
            
            # Final summary
            if all_valid:
                print("\nAll binary files are valid!")
            else:
                print("\nOne or more binary files failed validation. See details above.")
            
            return
    
    # Get binary file path
    binary_file = args.binary_file
    if not binary_file:
        if hasattr(config, 'binary') and hasattr(config.binary, 'train_file'):
            binary_file = config.binary.train_file
        else:
            binary_file = Path(config.data_dir) / "binary" / "train.h5"
    
    # Get metadata file path
    metadata_file = args.metadata_file
    if not metadata_file:
        metadata_file = Path(config.data_dir) / "train" / "metadata.json"
        if not metadata_file.exists():
            print(f"Metadata file not found at {metadata_file}, validation will proceed without comparison")
            metadata_file = None
    
    print(f"Validating binary file: {binary_file}")
    if metadata_file:
        print(f"Comparing with metadata file: {metadata_file}")
    
    # Validate the binary file
    validation_results = validate_binary_file(binary_file, metadata_file)
    
    # Print validation results
    print_validation_results(validation_results)


def print_validation_results(validation_results):
    """
    Print validation results in a formatted way
    
    Args:
        validation_results: Validation results dictionary
    """
    if validation_results['valid']:
        print("✅ Validation successful!")
        print("\nStatistics:")
        for key, value in validation_results.get('stats', {}).items():
            print(f"  • {key}: {value}")
    else:
        print("❌ Validation failed!")
        print("\nErrors:")
        for error in validation_results.get('errors', []):
            print(f"  • {error}")
    
    # Print warnings
    if validation_results.get('warnings', []):
        print("\nWarnings:")
        for warning in validation_results['warnings']:
            print(f"  • {warning}")


if __name__ == "__main__":
    main()