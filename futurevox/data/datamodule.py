import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from .datasets import FutureVoxDataset


class FutureVoxDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for FutureVox+
    
    This module handles:
    1. Dataset creation for training, validation, and testing
    2. DataLoader configuration
    3. Data preparation and setup
    4. Binary file loading support for improved performance
    """
    
    def __init__(self, config):
        """
        Initialize data module
        
        Args:
            config: Configuration object containing data parameters
        """
        super().__init__()
        self.config = config
        self.data_dir = Path(config.data_dir)
        
        # Save configuration for dataset initialization
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.val_batch_size = config.validation.batch_size
        self.val_num_workers = config.validation.num_workers
        
        # Binary file settings
        self.use_binary = config.binary.enabled if hasattr(config, 'binary') and hasattr(config.binary, 'enabled') else False
        
        if self.use_binary:
            self.binary_train = config.binary.train_file if hasattr(config.binary, 'train_file') else None
            self.binary_val = config.binary.val_file if hasattr(config.binary, 'val_file') else None
            self.binary_test = config.binary.test_file if hasattr(config.binary, 'test_file') else None
            
            # Validate binary files if specified
            if hasattr(config.binary, 'validate_on_load') and config.binary.validate_on_load:
                try:
                    from preprocessing.binary import validate_binary_file
                    
                    # Validate binary files if they exist
                    for file_attr, file_path in [
                        ('binary_train', self.binary_train),
                        ('binary_val', self.binary_val),
                        ('binary_test', self.binary_test)
                    ]:
                        if file_path and Path(file_path).exists():
                            print(f"Validating {file_attr}: {file_path}")
                            validation_result = validate_binary_file(file_path)
                            if not validation_result['valid']:
                                print(f"Warning: Binary file validation failed for {file_path}")
                                for error in validation_result['errors']:
                                    print(f"  Error: {error}")
                            else:
                                print(f"Binary file validation successful for {file_path}")
                                stats = validation_result.get('stats', {})
                                for key, value in stats.items():
                                    print(f"  {key}: {value}")
                except ImportError:
                    print("Warning: Binary validation module not available, skipping validation")
        else:
            self.binary_train = None
            self.binary_val = None
            self.binary_test = None
            
        # Audio preprocessing parameters
        self.sample_rate = config.preprocessing.audio.sample_rate
        self.n_fft = config.preprocessing.audio.n_fft
        self.hop_length = config.preprocessing.audio.hop_length
        self.win_length = config.preprocessing.audio.win_length
        self.n_mels = config.preprocessing.audio.n_mels
        self.f_min = config.preprocessing.audio.f_min
        self.f_max = config.preprocessing.audio.f_max
        
        # Text preprocessing parameters
        self.g2p_model = config.preprocessing.text.g2p_model
        self.phoneme_dict_path = config.preprocessing.text.phoneme_dict
        self.cleaner = config.preprocessing.text.cleaner
        
        # F0 parameters
        self.f0_min = config.preprocessing.f0.min_f0
        self.f0_max = config.preprocessing.f0.max_f0
        
        # Prepare paths
        self.train_path = self.data_dir / "train"
        self.val_path = self.data_dir / "val"
        self.test_path = self.data_dir / "test"
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Data preparation (download, preprocessing, etc.)
        This method is called only once and on only one GPU
        """
        # If binary files are enabled but not found, try to create them
        if self.use_binary:
            missing_binary = False
            for file_path in [self.binary_train, self.binary_val, self.binary_test]:
                if file_path and not Path(file_path).exists():
                    missing_binary = True
                    break
                    
            if missing_binary:
                try:
                    from preprocessing.binary import create_binary_splits
                    
                    # Check if metadata exists
                    metadata_file = self.train_path / "metadata.json"
                    if not metadata_file.exists():
                        print("Warning: Cannot create binary files, metadata not found")
                    else:
                        # Create binary files
                        binary_dir = Path(self.config.binary_dir) if hasattr(self.config, 'binary_dir') else self.data_dir / "binary"
                        binary_dir.mkdir(parents=True, exist_ok=True)
                        
                        print(f"Creating binary files in {binary_dir}")
                        create_binary_splits(
                            metadata_file, 
                            binary_dir, 
                            self.data_dir
                        )
                        
                        print("Binary files created")
                except ImportError:
                    print("Warning: Binary creation module not available")
                    
        # Check if data directories exist for standard loading
        if not self.use_binary and (not self.train_path.exists() or not self.val_path.exists()):
            raise FileNotFoundError(
                f"Data directories not found at {self.data_dir}. "
                "Please run preprocessing scripts first."
            )
    
    def setup(self, stage=None):
        """
        Data setup (create datasets)
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # Load phoneme dictionary
        phoneme_dict = {}
        if Path(self.phoneme_dict_path).exists():
            import json
            with open(self.phoneme_dict_path, 'r') as f:
                phoneme_dict = json.load(f)
        
        # Create datasets
        if stage == 'fit' or stage is None:
            print(f"Setting up training dataset{'(binary mode)' if self.use_binary else ''}")
            self.train_dataset = FutureVoxDataset(
                data_dir=self.train_path,
                phoneme_dict=phoneme_dict,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                g2p_model=self.g2p_model,
                cleaner=self.cleaner,
                binary_file=self.binary_train if self.use_binary else None
            )
            
            print(f"Setting up validation dataset{'(binary mode)' if self.use_binary else ''}")
            self.val_dataset = FutureVoxDataset(
                data_dir=self.val_path,
                phoneme_dict=phoneme_dict,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                g2p_model=self.g2p_model,
                cleaner=self.cleaner,
                binary_file=self.binary_val if self.use_binary else None
            )
        
        if stage == 'test' or stage is None:
            print(f"Setting up test dataset{'(binary mode)' if self.use_binary else ''}")
            self.test_dataset = FutureVoxDataset(
                data_dir=self.test_path,
                phoneme_dict=phoneme_dict,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
                f0_min=self.f0_min,
                f0_max=self.f0_max,
                g2p_model=self.g2p_model,
                cleaner=self.cleaner,
                binary_file=self.binary_test if self.use_binary else None
            )
    
    # The dataloader methods remain unchanged
    
    def train_dataloader(self):
        """
        Create training dataloader
        
        Returns:
            Training DataLoader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        """
        Create validation dataloader
        
        Returns:
            Validation DataLoader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        """
        Create test dataloader
        
        Returns:
            Test DataLoader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.val_num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        Custom collate function for batching
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched data
        """
        # Separate batch items
        phonemes = [item['phonemes'] for item in batch]
        durations = [item['durations'] for item in batch]
        f0 = [item['f0'] for item in batch]
        mel_spectrograms = [item['mel_spectrogram'] for item in batch]
        speaker_ids = [item['speaker_ids'] for item in batch]
        language_ids = [item['language_ids'] for item in batch]
        
        # Get max lengths
        max_phoneme_len = max(x.size(0) for x in phonemes)
        max_frame_len = max(x.size(0) for x in mel_spectrograms)
        
        # Pad phoneme sequences
        phoneme_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.long)
        for i, x in enumerate(phonemes):
            phoneme_padded[i, :x.size(0)] = x
        
        # Pad duration sequences
        duration_padded = torch.zeros(len(batch), max_phoneme_len, dtype=torch.float)
        for i, x in enumerate(durations):
            duration_padded[i, :x.size(0)] = x
        
        # Pad f0 sequences
        f0_padded = torch.zeros(len(batch), max_frame_len, 1, dtype=torch.float)
        for i, x in enumerate(f0):
            f0_padded[i, :x.size(0)] = x
        
        # Pad mel-spectrograms
        mel_padded = torch.zeros(len(batch), max_frame_len, mel_spectrograms[0].size(1), dtype=torch.float)
        for i, x in enumerate(mel_spectrograms):
            mel_padded[i, :x.size(0)] = x
        
        # Transpose mel-spectrograms for model input (batch, n_mels, time)
        mel_padded = mel_padded.transpose(1, 2)
        
        # Stack speaker and language IDs
        speaker_ids = torch.stack(speaker_ids)
        language_ids = torch.stack(language_ids)
        
        # Create batch dictionary
        batch_dict = {
            'phonemes': phoneme_padded,
            'durations': duration_padded,
            'f0': f0_padded,
            'mel_spectrogram': mel_padded,
            'speaker_ids': speaker_ids,
            'language_ids': language_ids
        }
        
        return batch_dict