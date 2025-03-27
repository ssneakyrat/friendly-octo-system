import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import json
import os
from pathlib import Path
import random

from utils.audio import compute_mel_spectrogram, extract_f0, load_audio

# Import the binary utilities
try:
    from preprocessing.binary import load_item_from_binary
except ImportError:
    # For backwards compatibility if the module doesn't exist
    def load_item_from_binary(*args, **kwargs):
        raise ImportError("Binary module not available")


class FutureVoxDataset(Dataset):
    """
    Dataset for FutureVox+ training and evaluation
    
    This dataset handles:
    1. Loading audio files, speaker information, and language information
    2. Computing mel-spectrograms and F0 contours
    3. Processing phoneme sequences and durations
    4. Generating speaker and language mixture vectors
    5. Loading from binary files for improved performance (if available)
    """
    
    def __init__(
        self,
        data_dir,
        phoneme_dict=None,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        f_min=0,
        f_max=8000,
        f0_min=65,
        f0_max=1000,
        g2p_model="espeak",
        cleaner="english_cleaners",
        binary_file=None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing data
            phoneme_dict: Dictionary mapping words to phonemes
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop size
            win_length: Window size
            n_mels: Number of mel bands
            f_min: Minimum frequency for mel-spectrogram
            f_max: Maximum frequency for mel-spectrogram
            f0_min: Minimum F0 frequency
            f0_max: Maximum F0 frequency
            g2p_model: Grapheme-to-phoneme model
            cleaner: Text cleaner
            binary_file: Path to binary dataset file (optional)
        """
        self.data_dir = Path(data_dir)
        self.phoneme_dict = phoneme_dict if phoneme_dict else {}
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.binary_file = binary_file
        self.use_binary = binary_file is not None
        
        # Load data
        if self.use_binary:
            self.load_from_binary()
        else:
            self.metadata_path = self.data_dir / "metadata.json"
            self.load_metadata()
        
        # Create phoneme vocabulary if not provided
        if not self.phoneme_dict:
            self.build_phoneme_vocabulary()
            
        # Create speaker and language ID maps
        self.build_speaker_language_maps()

    def load_from_binary(self):
        """
        Load metadata from binary file
        """
        if not Path(self.binary_file).exists():
            raise FileNotFoundError(f"Binary file not found at {self.binary_file}")
        
        try:
            with h5py.File(self.binary_file, 'r') as f:
                # Get number of items
                self.num_items = f.attrs['num_items']
                
                # Extract metadata if available
                if 'metadata' in f:
                    self.metadata = []
                    for i in range(int(self.num_items)):
                        try:
                            item = json.loads(f['metadata'][i])
                            self.metadata.append(item)
                        except:
                            # If metadata can't be loaded, create a placeholder
                            self.metadata.append({'idx': i})
                else:
                    # Create placeholder metadata
                    self.metadata = [{'idx': i} for i in range(int(self.num_items))]
                
                print(f"Loaded {len(self.metadata)} items from binary file {self.binary_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to load binary file: {str(e)}")
    
    def load_metadata(self):
        """
        Load metadata from JSON file
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter out entries with missing files
        valid_entries = []
        for entry in self.metadata:
            audio_path = self.data_dir / entry["audio_file"]
            if audio_path.exists():
                valid_entries.append(entry)
        
        self.metadata = valid_entries
    
    def build_phoneme_vocabulary(self):
        """
        Build phoneme vocabulary from metadata
        """
        # Import G2P model
        if self.g2p_model == "espeak":
            try:
                import phonemizer
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                
                self.phonemizer = phonemizer.backend.EspeakBackend(
                    language='en-us',
                    separator=Separator(phone=' ', syllable=None, word=None)
                )
            except ImportError:
                raise ImportError("Please install phonemizer: pip install phonemizer")
        else:
            raise ValueError(f"Unsupported G2P model: {self.g2p_model}")
        
        # Create phoneme set from all transcripts
        phoneme_set = set()
        for entry in self.metadata:
            if "phonemes" in entry:
                phonemes = entry["phonemes"].split()
                phoneme_set.update(phonemes)
            elif "text" in entry:
                phonemes = self.phonemizer.phonemize([entry["text"]], strip=True)[0].split()
                phoneme_set.update(phonemes)
        
        # Create phoneme-to-id mapping
        self.phoneme_dict = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        
        for i, phoneme in enumerate(sorted(phoneme_set)):
            self.phoneme_dict[phoneme] = i + 4
    
    def build_speaker_language_maps(self):
        """
        Build speaker and language ID maps
        """
        # Extract unique speakers and languages
        speakers = set()
        languages = set()
        
        for entry in self.metadata:
            if "speaker_id" in entry:
                speakers.add(entry["speaker_id"])
            if "language" in entry:
                languages.add(entry["language"])
        
        # Create ID mappings
        self.speaker_map = {speaker: i for i, speaker in enumerate(sorted(speakers))}
        self.language_map = {lang: i for i, lang in enumerate(sorted(languages))}
        
        self.num_speakers = len(self.speaker_map)
        self.num_languages = len(self.language_map)
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            Number of items in dataset
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - phonemes: Phoneme ID sequence
                - durations: Phoneme durations
                - f0: F0 contour
                - mel_spectrogram: Mel-spectrogram
                - speaker_ids: Speaker ID mixture
                - language_ids: Language ID mixture
        """
        if self.use_binary:
            # Load from binary file
            entry = load_item_from_binary(self.binary_file, idx)
        else:
            # Load from metadata
            entry = self.metadata[idx]
        
        # Process waveform - either load from file or use pre-loaded from binary
        if 'audio_waveform' in entry:
            # Waveform already loaded from binary
            waveform = entry['audio_waveform']
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
        else:
            # Load audio from file
            audio_path = self.data_dir / entry["audio_file"]
            waveform = load_audio(audio_path, self.sample_rate)
        
        # Compute mel-spectrogram
        mel_spectrogram = compute_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )
        
        # Transpose mel-spectrogram for easier handling (time, n_mels)
        mel_spectrogram = mel_spectrogram.squeeze(0).transpose(0, 1)
        
        # Extract F0
        f0 = extract_f0(
            waveform,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            f0_min=self.f0_min,
            f0_max=self.f0_max
        )
        
        # Reshape F0 to match mel-spectrogram length
        if f0.size(1) < mel_spectrogram.size(0):
            f0 = F.pad(f0, (0, 0, 0, mel_spectrogram.size(0) - f0.size(1)))
        elif f0.size(1) > mel_spectrogram.size(0):
            f0 = f0[:, :mel_spectrogram.size(0)]
        
        # Reshape to (time, 1)
        f0 = f0.squeeze(0).unsqueeze(1)
        
        # Process phonemes
        if "phonemes" in entry:
            phonemes = entry["phonemes"].split()
        else:
            phonemes = self.text_to_phonemes(entry["text"])
        
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        # Get durations
        if 'durations' in entry and isinstance(entry['durations'], torch.Tensor):
            # Durations already loaded from binary
            durations = entry['durations']
        elif "alignment_file" in entry:
            alignment_path = self.data_dir / entry["alignment_file"] if not self.use_binary else entry["alignment_file"]
            if os.path.exists(alignment_path):
                durations = self.get_duration_from_alignment(alignment_path)
            else:
                # If alignment not available, create uniform durations
                total_frames = mel_spectrogram.size(0)
                durations = torch.ones(len(phoneme_ids), dtype=torch.float) * (total_frames / len(phoneme_ids))
                # Adjust last duration to match exactly
                durations[-1] = total_frames - durations[:-1].sum()
        else:
            # Create uniform durations
            total_frames = mel_spectrogram.size(0)
            durations = torch.ones(len(phoneme_ids), dtype=torch.float) * (total_frames / len(phoneme_ids))
            # Adjust last duration to match exactly
            durations[-1] = total_frames - durations[:-1].sum()
        
        # Create speaker and language mixtures
        speaker_id = entry.get("speaker_id", "unknown")
        language = entry.get("language", "en")
        
        speaker_mixture = self.create_speaker_mixture(speaker_id)
        language_mixture = self.create_language_mixture(language)
        
        # Return item
        return {
            "phonemes": phoneme_ids,
            "durations": durations,
            "f0": f0,
            "mel_spectrogram": mel_spectrogram,
            "speaker_ids": speaker_mixture,
            "language_ids": language_mixture
        }

    # Other methods remain the same...
    
    def load_metadata(self):
        """
        Load metadata from JSON file
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
        
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Filter out entries with missing files
        valid_entries = []
        for entry in self.metadata:
            audio_path = self.data_dir / entry["audio_file"]
            if audio_path.exists():
                valid_entries.append(entry)
        
        self.metadata = valid_entries
    
    def build_phoneme_vocabulary(self):
        """
        Build phoneme vocabulary from metadata
        """
        # Import G2P model
        if self.g2p_model == "espeak":
            try:
                import phonemizer
                from phonemizer.backend import EspeakBackend
                from phonemizer.separator import Separator
                
                self.phonemizer = phonemizer.backend.EspeakBackend(
                    language='en-us',
                    separator=Separator(phone=' ', syllable=None, word=None)
                )
            except ImportError:
                raise ImportError("Please install phonemizer: pip install phonemizer")
        else:
            raise ValueError(f"Unsupported G2P model: {self.g2p_model}")
        
        # Create phoneme set from all transcripts
        phoneme_set = set()
        for entry in self.metadata:
            if "phonemes" in entry:
                phonemes = entry["phonemes"].split()
                phoneme_set.update(phonemes)
            elif "text" in entry:
                phonemes = self.phonemizer.phonemize([entry["text"]], strip=True)[0].split()
                phoneme_set.update(phonemes)
        
        # Create phoneme-to-id mapping
        self.phoneme_dict = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        
        for i, phoneme in enumerate(sorted(phoneme_set)):
            self.phoneme_dict[phoneme] = i + 4
    
    def build_speaker_language_maps(self):
        """
        Build speaker and language ID maps
        """
        # Extract unique speakers and languages
        speakers = set()
        languages = set()
        
        for entry in self.metadata:
            if "speaker_id" in entry:
                speakers.add(entry["speaker_id"])
            if "language" in entry:
                languages.add(entry["language"])
        
        # Create ID mappings
        self.speaker_map = {speaker: i for i, speaker in enumerate(sorted(speakers))}
        self.language_map = {lang: i for i, lang in enumerate(sorted(languages))}
        
        self.num_speakers = len(self.speaker_map)
        self.num_languages = len(self.language_map)
    
    def phonemes_to_ids(self, phonemes):
        """
        Convert phoneme sequence to IDs
        
        Args:
            phonemes: List of phonemes
            
        Returns:
            Tensor of phoneme IDs
        """
        # Convert phonemes to IDs
        phoneme_ids = [self.phoneme_dict.get(p, self.phoneme_dict["<unk>"]) for p in phonemes]
        
        # Add BOS and EOS tokens
        phoneme_ids = [self.phoneme_dict["<bos>"]] + phoneme_ids + [self.phoneme_dict["<eos>"]]
        
        return torch.tensor(phoneme_ids, dtype=torch.long)
    
    def get_duration_from_alignment(self, alignment_file):
        """
        Get phoneme durations from alignment file
        
        Args:
            alignment_file: Path to alignment file
            
        Returns:
            Tensor of phoneme durations
        """
        try:
            # Load alignment
            with open(alignment_file, 'r') as f:
                alignment = json.load(f)
            
            # Extract durations (convert from indices to frame counts)
            durations = []
            prev_idx = 0
            for idx in alignment:
                durations.append(idx - prev_idx)
                prev_idx = idx
            
            return torch.tensor(durations, dtype=torch.float)
        except (FileNotFoundError, json.JSONDecodeError):
            # If alignment file doesn't exist, return dummy durations
            return torch.ones(10, dtype=torch.float)  # Default duration
    
    def create_speaker_mixture(self, primary_speaker_id, num_mixtures=2):
        """
        Create random speaker mixture
        
        Args:
            primary_speaker_id: Primary speaker ID
            num_mixtures: Number of speakers to mix
            
        Returns:
            Tensor of speaker ID mixtures [num_mixtures, 2]
            Each row contains [speaker_id, weight]
        """
        # Convert primary speaker ID to internal ID
        primary_id = self.speaker_map.get(primary_speaker_id, 0)
        
        # Initialize mixture tensor
        speaker_mixture = torch.zeros(num_mixtures, 2)
        
        # Set primary speaker
        speaker_mixture[0, 0] = primary_id
        speaker_mixture[0, 1] = 0.7  # Primary speaker weight
        
        # Add secondary speakers if available
        if self.num_speakers > 1:
            available_speakers = list(range(self.num_speakers))
            if primary_id in available_speakers:
                available_speakers.remove(primary_id)
            
            # Select random secondary speakers
            secondary_speakers = random.sample(
                available_speakers, 
                min(num_mixtures - 1, len(available_speakers))
            )
            
            # Assign random weights
            for i, speaker_id in enumerate(secondary_speakers, 1):
                if i < num_mixtures:
                    speaker_mixture[i, 0] = speaker_id
                    speaker_mixture[i, 1] = 0.3 / (num_mixtures - 1)  # Split remaining weight
        
        return speaker_mixture
    
    def create_language_mixture(self, primary_language, num_mixtures=2):
        """
        Create random language mixture
        
        Args:
            primary_language: Primary language
            num_mixtures: Number of languages to mix
            
        Returns:
            Tensor of language ID mixtures [num_mixtures, 2]
            Each row contains [language_id, weight]
        """
        # Convert primary language to internal ID
        primary_id = self.language_map.get(primary_language, 0)
        
        # Initialize mixture tensor
        language_mixture = torch.zeros(num_mixtures, 2)
        
        # Set primary language
        language_mixture[0, 0] = primary_id
        language_mixture[0, 1] = 0.8  # Primary language weight
        
        # Add secondary languages if available
        if self.num_languages > 1:
            available_languages = list(range(self.num_languages))
            if primary_id in available_languages:
                available_languages.remove(primary_id)
            
            # Select random secondary languages
            secondary_languages = random.sample(
                available_languages, 
                min(num_mixtures - 1, len(available_languages))
            )
            
            # Assign random weights
            for i, language_id in enumerate(secondary_languages, 1):
                if i < num_mixtures:
                    language_mixture[i, 0] = language_id
                    language_mixture[i, 1] = 0.2 / (num_mixtures - 1)  # Split remaining weight
        
        return language_mixture
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            Number of items in dataset
        """
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing:
                - phonemes: Phoneme ID sequence
                - durations: Phoneme durations
                - f0: F0 contour
                - mel_spectrogram: Mel-spectrogram
                - speaker_ids: Speaker ID mixture
                - language_ids: Language ID mixture
        """
        entry = self.metadata[idx]
        
        # Load audio
        audio_path = self.data_dir / entry["audio_file"]
        waveform = load_audio(audio_path, self.sample_rate)
        
        # Compute mel-spectrogram
        mel_spectrogram = compute_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        )
        
        # Transpose mel-spectrogram for easier handling (time, n_mels)
        mel_spectrogram = mel_spectrogram.squeeze(0).transpose(0, 1)
        
        # Extract F0
        f0 = extract_f0(
            waveform,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            f0_min=self.f0_min,
            f0_max=self.f0_max
        )
        
        # Reshape F0 to match mel-spectrogram length
        if f0.size(1) < mel_spectrogram.size(0):
            f0 = F.pad(f0, (0, 0, 0, mel_spectrogram.size(0) - f0.size(1)))
        elif f0.size(1) > mel_spectrogram.size(0):
            f0 = f0[:, :mel_spectrogram.size(0)]
        
        # Reshape to (time, 1)
        f0 = f0.squeeze(0).unsqueeze(1)
        
        # Process phonemes
        # Process phonemes - only from .lab files
        if "phonemes" in entry:
            phonemes = entry["phonemes"].split()
        else:
            raise ValueError(f"No phonemes found for entry {idx}. All entries must have phonemes from .lab files.")

        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        # Get durations
        if "alignment_file" in entry and os.path.exists(self.data_dir / entry["alignment_file"]):
            durations = self.get_duration_from_alignment(self.data_dir / entry["alignment_file"])
        else:
            # If alignment not available, create uniform durations
            total_frames = mel_spectrogram.size(0)
            durations = torch.ones(len(phoneme_ids), dtype=torch.float) * (total_frames / len(phoneme_ids))
            # Adjust last duration to match exactly
            durations[-1] = total_frames - durations[:-1].sum()
        
        # Create speaker and language mixtures
        speaker_id = entry.get("speaker_id", "unknown")
        language = entry.get("language", "en")
        
        speaker_mixture = self.create_speaker_mixture(speaker_id)
        language_mixture = self.create_language_mixture(language)
        
        # Return item
        return {
            "phonemes": phoneme_ids,
            "durations": durations,
            "f0": f0,
            "mel_spectrogram": mel_spectrogram,
            "speaker_ids": speaker_mixture,
            "language_ids": language_mixture
        }