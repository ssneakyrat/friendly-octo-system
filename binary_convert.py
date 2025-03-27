#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import yaml
import os
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf

# Import binary utilities
try:
    from preprocessing.binary import save_dataset_to_binary, validate_binary_file, create_binary_splits
except ImportError:
    raise ImportError("Binary utilities not found. Make sure preprocessing/binary.py exists.")


def convert_metadata_to_binary(metadata_file, output_path, data_dir=None):
    """
    Convert a metadata file to binary format
    
    Args:
        metadata_file: Path to metadata file
        output_path: Path to save binary file
        data_dir: Base directory for resolving relative paths
        
    Returns:
        Path to created binary file
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Converting {len(metadata)} entries to binary format")
    
    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to binary
    binary_path = save_dataset_to_binary(metadata, output_path, data_dir)
    
    print(f"Binary file saved to {binary_path}")
    
    return binary_path


def convert_all_splits(data_dir, binary_dir, create_missing_splits=False):
    """
    Convert all dataset splits to binary format
    
    Args:
        data_dir: Base data directory
        binary_dir: Directory to save binary files
        create_missing_splits: Whether to create missing splits
        
    Returns:
        Dictionary of binary file paths
    """
    data_dir = Path(data_dir)
    binary_dir = Path(binary_dir)
    binary_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if splits exist
    train_metadata = data_dir / "train" / "metadata.json"
    val_metadata = data_dir / "val" / "metadata.json"
    test_metadata = data_dir / "test" / "metadata.json"
    
    # Check if train split exists
    if not train_metadata.exists():
        raise FileNotFoundError(f"Training metadata not found at {train_metadata}")
    
    # If validation and test splits don't exist, create them if requested
    if create_missing_splits and (not val_metadata.exists() or not test_metadata.exists()):
        from preprocess import create_val_test_split
        print("Creating validation and test splits")
        create_val_test_split(train_metadata)
    
    # Convert each split
    binary_files = {}
    
    # Convert train split
    if train_metadata.exists():
        print("\nConverting training data:")
        train_binary = binary_dir / "train.h5"
        convert_metadata_to_binary(train_metadata, train_binary, data_dir)
        binary_files['train'] = train_binary
    
    # Convert validation split
    if val_metadata.exists():
        print("\nConverting validation data:")
        val_binary = binary_dir / "val.h5"
        convert_metadata_to_binary(val_metadata, val_binary, data_dir)
        binary_files['val'] = val_binary
    
    # Convert test split
    if test_metadata.exists():
        print("\nConverting test data:")
        test_binary = binary_dir / "test.h5"
        convert_metadata_to_binary(test_metadata, test_binary, data_dir)
        binary_files['test'] = test_binary
    
    return binary_files


def main():
    """
    Convert FutureVox+ dataset to binary format
    """
    parser = argparse.ArgumentParser(description="Convert FutureVox+ dataset to binary format")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Base data directory (overrides config)")
    parser.add_argument("--binary_dir", type=str, help="Directory to save binary files (overrides config)")
    parser.add_argument("--metadata_file", type=str, help="Path to specific metadata file to convert")
    parser.add_argument("--output_path", type=str, help="Path to save specific binary file")
    parser.add_argument("--create_missing_splits", action="store_true", help="Create missing splits if they don't exist")
    parser.add_argument("--validate", action="store_true", help="Validate created binary files")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = OmegaConf.create(config)
    
    # If specific metadata file is provided, convert just that file
    if args.metadata_file:
        metadata_file = Path(args.metadata_file)
        output_path = args.output_path or Path(args.binary_dir or config.binary_dir) / f"{metadata_file.stem}.h5"
        
        # Get base directory
        data_dir = args.data_dir or config.data_dir
        
        # Convert to binary
        binary_path = convert_metadata_to_binary(metadata_file, output_path, data_dir)
        
        # Validate if requested
        if args.validate:
            print("\nValidating binary file:")
            validation_result = validate_binary_file(binary_path, metadata_file)
            
            if validation_result['valid']:
                print(f"✅ Binary file validation successful!")
                for key, value in validation_result.get('stats', {}).items():
                    print(f"  • {key}: {value}")
            else:
                print(f"❌ Binary file validation failed!")
                for error in validation_result.get('errors', []):
                    print(f"  • {error}")
        
        return
    
    # Convert all splits
    data_dir = args.data_dir or config.data_dir
    binary_dir = args.binary_dir or config.binary_dir
    
    print(f"Converting all dataset splits from {data_dir} to binary format in {binary_dir}")
    
    binary_files = convert_all_splits(data_dir, binary_dir, args.create_missing_splits)
    
    # Validate if requested
    if args.validate:
        print("\n=== Validating Binary Files ===")
        all_valid = True
        
        for split_name, binary_path in binary_files.items():
            print(f"\nValidating {split_name} binary file: {binary_path}")
            
            # Find corresponding metadata file
            metadata_file = Path(data_dir) / split_name / "metadata.json"
            
            validation_result = validate_binary_file(binary_path, metadata_file)
            all_valid = all_valid and validation_result['valid']
            
            if validation_result['valid']:
                print(f"✅ {split_name.capitalize()} binary file validation successful!")
                for key, value in validation_result.get('stats', {}).items():
                    print(f"  • {key}: {value}")
            else:
                print(f"❌ {split_name.capitalize()} binary file validation failed!")
                for error in validation_result.get('errors', []):
                    print(f"  • {error}")
        
        # Final summary
        print("\n=== Validation Summary ===")
        if all_valid:
            print("✅ All binary files are valid!")
        else:
            print("❌ One or more binary files failed validation. See details above.")
    
    # Update configuration file if all files were created successfully
    if len(binary_files) > 0:
        print("\nUpdating configuration to use binary files:")
        
        # Update binary settings in config
        if not hasattr(config, 'binary'):
            config.binary = {}
        
        config.binary.enabled = True
        
        if 'train' in binary_files:
            config.binary.train_file = str(binary_files['train'])
            print(f"  • Training binary: {binary_files['train']}")
        
        if 'val' in binary_files:
            config.binary.val_file = str(binary_files['val'])
            print(f"  • Validation binary: {binary_files['val']}")
        
        if 'test' in binary_files:
            config.binary.test_file = str(binary_files['test'])
            print(f"  • Test binary: {binary_files['test']}")
        
        # Save updated config
        updated_config_path = Path(args.config).with_name(f"{Path(args.config).stem}_binary.yaml")
        OmegaConf.save(config, updated_config_path)
        
        print(f"\nUpdated configuration saved to {updated_config_path}")
        print("You can use this config file with train.py to use binary files for training.")


if __name__ == "__main__":
    main()