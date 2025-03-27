import h5py
import torch
import torchaudio
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def save_dataset_to_binary(metadata, output_path, data_dir=None, sample_rate=22050):
    """
    Save dataset to a binary HDF5 file for more efficient loading
    
    Args:
        metadata: List of metadata entries
        output_path: Path to save binary file
        data_dir: Base directory for resolving relative paths
        sample_rate: Audio sample rate
        
    Returns:
        Path to created binary file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create HDF5 file
    with h5py.File(output_path, 'w') as f:
        # Create groups
        audio_group = f.create_group('audio')
        mel_group = f.create_group('mel_spectrograms')
        f0_group = f.create_group('f0')
        phoneme_group = f.create_group('phonemes')
        duration_group = f.create_group('durations')
        speaker_group = f.create_group('speaker_ids')
        language_group = f.create_group('language_ids')
        
        # Store metadata
        f.attrs['num_items'] = len(metadata)
        
        # Create string dataset type for variable-length strings
        dt = h5py.special_dtype(vlen=str)
        
        # Store original metadata as JSON
        metadata_ds = f.create_dataset('metadata', (len(metadata),), dtype=dt)
        for i, item in enumerate(metadata):
            metadata_ds[i] = json.dumps(item)
        
        # Store all items
        for i, item in enumerate(tqdm(metadata, desc="Saving to binary")):
            item_id = f"item_{i:08d}"
            
            # Store audio file path and content
            audio_path = item['audio_file']
            if data_dir is not None:
                full_audio_path = Path(data_dir) / audio_path
            else:
                full_audio_path = Path(audio_path)
            
            try:
                # Load audio
                waveform, sr = torchaudio.load(full_audio_path)
                
                # Store audio data
                audio_group.create_dataset(
                    item_id, 
                    data=waveform.numpy(), 
                    compression="gzip", 
                    compression_opts=4
                )
                
                # Store sample rate
                audio_group[item_id].attrs['sample_rate'] = sr
                audio_group[item_id].attrs['original_path'] = str(audio_path)
            except Exception as e:
                print(f"Warning: Failed to load audio {full_audio_path}: {e}")
                audio_group.create_dataset(item_id, data=str(audio_path), dtype=dt)
            
            # Store phonemes
            if 'phonemes' in item:
                phoneme_list = item['phonemes'].split()
                # Store as a single string instead of an array
                phoneme_str = ' '.join(phoneme_list)
                phoneme_group.create_dataset(item_id, data=phoneme_str, dtype=dt)
            
            # Store speaker and language
            speaker_id = item.get('speaker_id', 'unknown')
            language_id = item.get('language', 'en')
            
            speaker_group.create_dataset(item_id, data=speaker_id, dtype=dt)
            language_group.create_dataset(item_id, data=language_id, dtype=dt)
            
            # Store alignment data if available
            if 'alignment_file' in item:
                alignment_path = item['alignment_file']
                if data_dir is not None:
                    full_alignment_path = Path(data_dir) / alignment_path
                else:
                    full_alignment_path = Path(alignment_path)
                
                try:
                    # Load alignment data
                    with open(full_alignment_path, 'r') as af:
                        alignment_data = json.load(af)
                    
                    # Store alignment data
                    duration_group.create_dataset(
                        item_id, 
                        data=np.array(alignment_data, dtype=np.float32),
                        compression="gzip", 
                        compression_opts=4
                    )
                    duration_group[item_id].attrs['original_path'] = str(alignment_path)
                except Exception as e:
                    print(f"Warning: Failed to load alignment {full_alignment_path}: {e}")
                    duration_group.create_dataset(item_id, data=str(alignment_path), dtype=dt)
    
    return output_path

def validate_binary_file(binary_path, metadata_path=None):
    """
    Validate binary dataset file
    
    Args:
        binary_path: Path to binary file
        metadata_path: Path to original metadata file (optional, for comparison)
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    binary_path = Path(binary_path)
    
    if not binary_path.exists():
        validation_results['errors'].append(f"Binary file does not exist: {binary_path}")
        return validation_results
    
    try:
        with h5py.File(binary_path, 'r') as f:
            # Basic structure checks
            required_groups = ['audio', 'phonemes', 'speaker_ids', 'language_ids']
            for group in required_groups:
                if group not in f:
                    validation_results['errors'].append(f"Missing required group: {group}")
            
            # Check if metadata is present
            if 'metadata' not in f:
                validation_results['errors'].append("Missing metadata dataset")
            
            # Check item count
            if 'num_items' not in f.attrs:
                validation_results['errors'].append("Missing num_items attribute")
            else:
                num_items = f.attrs['num_items']
                validation_results['stats']['num_items'] = num_items
                
                # Check that a few random items exist
                import random
                sample_indices = random.sample(range(int(num_items)), min(5, int(num_items)))
                
                for idx in sample_indices:
                    item_id = f"item_{idx:08d}"
                    
                    for group in required_groups:
                        if group in f and item_id not in f[group]:
                            validation_results['errors'].append(f"Item {item_id} missing from group {group}")
            
            # Compare with original metadata if provided
            if metadata_path is not None:
                try:
                    with open(metadata_path, 'r') as mf:
                        original_metadata = json.load(mf)
                        
                        # Check item count
                        if len(original_metadata) != num_items:
                            validation_results['errors'].append(
                                f"Original metadata has {len(original_metadata)} items, but binary file has {num_items}"
                            )
                        
                        # Check a few random items for consistency
                        sample_indices = random.sample(
                            range(min(len(original_metadata), int(num_items))), 
                            min(5, len(original_metadata))
                        )
                        
                        for idx in sample_indices:
                            orig_item = original_metadata[idx]
                            item_id = f"item_{idx:08d}"
                            
                            # Check speaker ID
                            if item_id in f['speaker_ids']:
                                binary_speaker = f['speaker_ids'][item_id][()]
                                if binary_speaker != orig_item.get('speaker_id', 'unknown'):
                                    validation_results['errors'].append(
                                        f"Speaker ID mismatch for item {idx}: {binary_speaker} vs {orig_item.get('speaker_id', 'unknown')}"
                                    )
                            
                            # Check language
                            if item_id in f['language_ids']:
                                binary_language = f['language_ids'][item_id][()]
                                if binary_language != orig_item.get('language', 'en'):
                                    validation_results['errors'].append(
                                        f"Language mismatch for item {idx}: {binary_language} vs {orig_item.get('language', 'en')}"
                                    )
                except Exception as e:
                    validation_results['errors'].append(f"Failed to compare with original metadata: {str(e)}")
            
            # Get statistical information
            speakers = set()
            languages = set()
            
            sample_size = min(100, int(num_items))
            sample_indices = random.sample(range(int(num_items)), sample_size)
            
            for idx in sample_indices:
                item_id = f"item_{idx:08d}"
                
                if 'speaker_ids' in f and item_id in f['speaker_ids']:
                    speakers.add(f['speaker_ids'][item_id][()])
                
                if 'language_ids' in f and item_id in f['language_ids']:
                    languages.add(f['language_ids'][item_id][()])
            
            validation_results['stats']['unique_speakers'] = len(speakers)
            validation_results['stats']['unique_languages'] = len(languages)
        
        # Set validation status
        validation_results['valid'] = len(validation_results['errors']) == 0
        
    except Exception as e:
        validation_results['errors'].append(f"Failed to open binary file: {str(e)}")
        validation_results['valid'] = False
    
    return validation_results

def load_item_from_binary(binary_path, item_idx):
    """
    Load a single item from binary file
    
    Args:
        binary_path: Path to binary file
        item_idx: Item index
        
    Returns:
        Dictionary with item data
    """
    item_data = {}
    
    with h5py.File(binary_path, 'r') as f:
        item_id = f"item_{item_idx:08d}"
        
        # Load audio data
        if 'audio' in f and item_id in f['audio']:
            # Check if stored as dataset or path
            audio_ds = f['audio'][item_id]
            if isinstance(audio_ds[()], str):
                item_data['audio_file'] = audio_ds[()]
            else:
                item_data['audio_waveform'] = torch.from_numpy(audio_ds[()])
                item_data['sample_rate'] = audio_ds.attrs.get('sample_rate', 22050)
                item_data['audio_file'] = audio_ds.attrs.get('original_path', '')
        
        # Load phonemes
        if 'phonemes' in f and item_id in f['phonemes']:
            phoneme_str = f['phonemes'][item_id][()]
            # If it's a byte string, decode it
            if isinstance(phoneme_str, bytes):
                phoneme_str = phoneme_str.decode('utf-8')
            item_data['phonemes'] = phoneme_str
        
        # Load speaker and language
        if 'speaker_ids' in f and item_id in f['speaker_ids']:
            item_data['speaker_id'] = f['speaker_ids'][item_id][()]
        
        if 'language_ids' in f and item_id in f['language_ids']:
            item_data['language'] = f['language_ids'][item_id][()]
        
        # Load durations
        if 'durations' in f and item_id in f['durations']:
            # Check if stored as dataset or path
            duration_ds = f['durations'][item_id]
            if isinstance(duration_ds[()], str):
                item_data['alignment_file'] = duration_ds[()]
            else:
                item_data['durations'] = torch.from_numpy(duration_ds[()])
                item_data['alignment_file'] = duration_ds.attrs.get('original_path', '')
        
        # Load metadata if available
        if 'metadata' in f and item_idx < len(f['metadata']):
            try:
                metadata = json.loads(f['metadata'][item_idx])
                # Update item_data with metadata fields
                for key, value in metadata.items():
                    if key not in item_data:
                        item_data[key] = value
            except:
                pass
    
    return item_data

def create_binary_splits(metadata_file, output_dir, data_dir=None, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    Create binary files for train/val/test splits
    
    Args:
        metadata_file: Path to metadata file
        output_dir: Directory to save binary files
        data_dir: Base directory for resolving relative paths
        val_ratio: Validation ratio
        test_ratio: Test ratio
        random_seed: Random seed
        
    Returns:
        Paths to created binary files
    """
    import random
    random.seed(random_seed)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Save binary files
    train_binary = output_dir / "train.h5"
    val_binary = output_dir / "val.h5"
    test_binary = output_dir / "test.h5"
    
    # Save train binary
    print(f"Creating training binary file with {len(train_data)} entries")
    save_dataset_to_binary(train_data, train_binary, data_dir)
    
    # Save validation binary
    print(f"Creating validation binary file with {len(val_data)} entries")
    save_dataset_to_binary(val_data, val_binary, data_dir)
    
    # Save test binary
    print(f"Creating test binary file with {len(test_data)} entries")
    save_dataset_to_binary(test_data, test_binary, data_dir)
    
    return {
        'train': train_binary,
        'val': val_binary,
        'test': test_binary
    }