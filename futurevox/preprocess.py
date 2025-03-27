#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
from pathlib import Path
import torchaudio
import torch
import numpy as np
from tqdm import tqdm
import shutil
import yaml
from omegaconf import OmegaConf

try:
    from preprocessing.binary import save_dataset_to_binary, validate_binary_file, create_binary_splits
    BINARY_SUPPORT = True
except ImportError:
    BINARY_SUPPORT = False
    print("Warning: Binary file support not available, install h5py for binary file support")


def parse_lab_file(lab_file_path, hop_length=256, sample_rate=22050, time_unit="samples"):
    """
    Parse .lab file and convert to frame indices and phonemes
    
    Args:
        lab_file_path: Path to .lab file
        hop_length: Hop length for spectrogram
        sample_rate: Audio sample rate
        time_unit: Unit of time in .lab file ('samples', 'seconds', '100ns')
        
    Returns:
        List of frame indices, List of phonemes
    """
    frame_indices = []
    phonemes = []
    
    with open(lab_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                phoneme = parts[2]
                
                # Convert time to samples based on unit
                if time_unit == "seconds":
                    end_sample = int(end_time * sample_rate)
                elif time_unit == "100ns":
                    end_sample = int(end_time * sample_rate / 10000000)
                else:  # Assume samples
                    end_sample = int(end_time)
                
                # Convert samples to frame index
                end_frame = end_sample // hop_length
                
                frame_indices.append(end_frame)
                phonemes.append(phoneme)
    
    return frame_indices, phonemes


def create_alignment_file(frame_indices, output_path):
    """
    Create alignment file in the format expected by FutureVoxDataset
    
    Args:
        frame_indices: List of frame indices
        output_path: Path to save the alignment file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(frame_indices, f)


def process_dataset(data_dir, output_dir, hop_length=256, sample_rate=22050, time_unit="samples", create_binary=False):
    """
    Process dataset: find .lab and .wav files, create alignments and metadata
    
    Args:
        data_dir: Directory containing speaker folders
        output_dir: Directory to save processed data
        hop_length: Hop length for spectrogram
        sample_rate: Audio sample rate
        time_unit: Unit of time in .lab file ('samples', 'seconds', '100ns')
        create_binary: Whether to create binary dataset files
        
    Returns:
        Path to generated metadata file
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directory structure
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = processed_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    # Create metadata list
    metadata = []
    
    # Find all speaker directories
    speaker_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    print(f"Found {len(speaker_dirs)} speaker directories")
    
    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = speaker_dir.name
        
        # Create speaker output directory
        speaker_output_dir = train_dir / speaker_id
        speaker_output_dir.mkdir(exist_ok=True)
        
        # Find all language directories for this speaker
        language_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
        
        if not language_dirs:
            print(f"Warning: No language directories found for speaker {speaker_id}")
            
            # Fallback to finding files directly in the speaker directory
            lab_files = list(speaker_dir.glob("*.lab"))
            if lab_files:
                print(f"  Found {len(lab_files)} .lab files directly in speaker directory")
                process_files(lab_files, speaker_id, "en", speaker_output_dir, train_dir, 
                              hop_length, sample_rate, time_unit, metadata)
        else:
            print(f"Found {len(language_dirs)} language directories for speaker {speaker_id}")
            
            # Process each language directory
            for language_dir in language_dirs:
                language_id = language_dir.name
                
                # Find all .lab files in this language directory
                lab_files = list(language_dir.glob("*.lab"))
                
                print(f"  Found {len(lab_files)} .lab files for language {language_id}")
                
                # Process files for this language
                process_files(lab_files, speaker_id, language_id, speaker_output_dir, train_dir, 
                             hop_length, sample_rate, time_unit, metadata)
    
    # Save metadata file
    metadata_file = train_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Processed {len(metadata)} files. Metadata saved to {metadata_file}")
    
    # Create binary files if requested and supported
    if create_binary and BINARY_SUPPORT:
        binary_dir = processed_dir / "binary"
        binary_dir.mkdir(exist_ok=True)
        
        print(f"Creating binary files in {binary_dir}")
        binary_files = create_binary_splits(
            metadata_file, 
            binary_dir,
            train_dir
        )
        
        # Validate binary files
        for split_name, binary_path in binary_files.items():
            print(f"Validating {split_name} binary file: {binary_path}")
            validation_result = validate_binary_file(binary_path, 
                                                   metadata_file if split_name == 'train' else None)
            
            if validation_result['valid']:
                print(f"✅ {split_name.capitalize()} binary file validation successful!")
                for key, value in validation_result.get('stats', {}).items():
                    print(f"  • {key}: {value}")
            else:
                print(f"❌ {split_name.capitalize()} binary file validation failed!")
                for error in validation_result.get('errors', []):
                    print(f"  • {error}")
    
    return metadata_file


def process_files(lab_files, speaker_id, language_id, speaker_output_dir, train_dir, 
                 hop_length, sample_rate, time_unit, metadata):
    """
    Process a list of .lab files
    
    Args:
        lab_files: List of .lab file paths
        speaker_id: Speaker ID
        language_id: Language ID
        speaker_output_dir: Output directory for this speaker
        train_dir: Training data directory (for relative paths)
        hop_length: Hop length for spectrogram
        sample_rate: Audio sample rate
        time_unit: Unit of time in .lab file
        metadata: Metadata list to append entries to
    """
    for lab_file in lab_files:
        # Find corresponding .wav file
        wav_file = lab_file.with_suffix(".wav")
        
        if not wav_file.exists():
            print(f"Warning: No WAV file found for {lab_file}")
            continue
        
        # Create language subdirectory in speaker output directory
        language_output_dir = speaker_output_dir / language_id
        language_output_dir.mkdir(exist_ok=True)
        
        # Copy .wav file to output directory with language subdirectory
        output_wav_file = language_output_dir / wav_file.name
        
        try:
            # Load and potentially resample audio
            waveform, sr = torchaudio.load(wav_file)
            
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
            # Save processed audio
            torchaudio.save(output_wav_file, waveform, sample_rate)
        except Exception as e:
            print(f"Error processing audio file {wav_file}: {e}")
            # Just copy the file directly if error occurs
            shutil.copy(wav_file, output_wav_file)
        
        # Parse .lab file to get phonemes and frame indices
        frame_indices, phonemes = parse_lab_file(lab_file, hop_length, sample_rate, time_unit)
        
        # Create alignment file
        alignment_dir = language_output_dir / "alignments"
        alignment_dir.mkdir(exist_ok=True)
        alignment_file = alignment_dir / f"{lab_file.stem}_alignment.json"
        create_alignment_file(frame_indices, alignment_file)
        
        # Create metadata entry
        rel_wav_path = output_wav_file.relative_to(train_dir)
        rel_alignment_path = alignment_file.relative_to(train_dir)
        
        entry = {
            "audio_file": str(rel_wav_path),
            "speaker_id": speaker_id,
            "phonemes": " ".join(phonemes),
            "alignment_file": str(rel_alignment_path),
            "language": language_id
        }
        metadata.append(entry)


def build_phoneme_vocabulary(metadata_file):
    """
    Build phoneme vocabulary from metadata
    
    Args:
        metadata_file: Path to metadata file
        
    Returns:
        Phoneme vocabulary dictionary
    """
    # Initialize vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3
    }
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Collect phonemes
    phoneme_set = set()
    for entry in metadata:
        if "phonemes" in entry:
            phonemes = entry["phonemes"].split()
            phoneme_set.update(phonemes)
    
    # Add phonemes to vocabulary
    for i, phoneme in enumerate(sorted(phoneme_set)):
        vocab[phoneme] = i + 4
    
    # Save vocabulary
    vocab_file = Path(metadata_file).parent / "phoneme_dict.json"
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Found {len(phoneme_set)} unique phonemes")
    print(f"Phoneme vocabulary saved to {vocab_file}")
    return vocab


def create_val_test_split(metadata_file, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create validation and test splits from training data
    
    Args:
        metadata_file: Path to metadata file
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    """
    import random
    random.seed(random_seed)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Group by speaker to maintain speaker balance
    speaker_groups = {}
    for entry in metadata:
        speaker_id = entry["speaker_id"]
        if speaker_id not in speaker_groups:
            speaker_groups[speaker_id] = []
        speaker_groups[speaker_id].append(entry)
    
    # Create splits
    train_data = []
    val_data = []
    test_data = []
    
    for speaker_id, entries in speaker_groups.items():
        # Shuffle entries for this speaker
        random.shuffle(entries)
        
        # Calculate split sizes
        total = len(entries)
        val_size = max(1, int(total * val_ratio))
        test_size = max(1, int(total * test_ratio))
        train_size = total - val_size - test_size
        
        # Split data
        train_data.extend(entries[:train_size])
        val_data.extend(entries[train_size:train_size + val_size])
        test_data.extend(entries[train_size + val_size:])
    
    # Create directories
    metadata_dir = Path(metadata_file).parent.parent
    train_dir = metadata_dir / "train"
    val_dir = metadata_dir / "val"
    test_dir = metadata_dir / "test"
    
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    # Save splits
    with open(train_dir / "metadata.json", 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_dir / "metadata.json", 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(test_dir / "metadata.json", 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")


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


def main():
    parser = argparse.ArgumentParser(description="Process dataset for FutureVox training")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                        help="Path to configuration file (will use paths defined there)")
    parser.add_argument("--data_dir", type=str, help="Directory containing speaker folders (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Directory to save processed data (overrides config)")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length for spectrogram")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--time_unit", type=str, default="samples", 
                        choices=["samples", "seconds", "100ns"],
                        help="Time unit in .lab files")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data splitting")
    parser.add_argument("--create_binary", action="store_true", help="Create binary dataset files")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Get settings from config or arguments
    data_dir = args.data_dir or (config.data_dir if config else "data/raw")
    output_dir = args.output_dir or (config.output_dir if config else "outputs")
    sample_rate = args.sample_rate or (config.preprocessing.audio.sample_rate if config else 22050)
    hop_length = args.hop_length or (config.preprocessing.audio.hop_length if config else 256)
    create_binary = args.create_binary or (config.binary.enabled if hasattr(config, 'binary') and hasattr(config.binary, 'enabled') else False)
    
    # Process dataset
    metadata_file = process_dataset(
        data_dir, 
        output_dir, 
        hop_length, 
        sample_rate,
        args.time_unit,
        create_binary
    )
    
    # Build phoneme vocabulary
    build_phoneme_vocabulary(metadata_file)
    
    # Create validation and test splits
    create_val_test_split(metadata_file, args.val_ratio, args.test_ratio, args.random_seed)
    
    print("Dataset preparation completed!")


if __name__ == "__main__":
    main()