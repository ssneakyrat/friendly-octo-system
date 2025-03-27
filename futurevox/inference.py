#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Qt
import matplotlib.pyplot as plt
from pathlib import Path
from omegaconf import OmegaConf
import json
from tqdm import tqdm

from models.futurevox import FutureVoxModel
from utils.audio import save_audio, plot_mel_spectrogram
from utils.visualization import plot_attention, plot_f0_contour
from preprocessing.text import text_to_phonemes


def load_config(config_path):
    """
    Load configuration file
    
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


def load_model(config, checkpoint_path):
    """
    Load trained model from checkpoint
    
    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Loaded model
    """
    # If checkpoint path is "best" or "last", find the appropriate file
    if checkpoint_path == "best" or checkpoint_path == "last":
        checkpoint_dir = Path(config.checkpoint_dir)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        if checkpoint_path == "best":
            # Find checkpoint with lowest validation loss
            checkpoints = list(checkpoint_dir.glob("futurevox-*.ckpt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            
            # Sort by validation loss (embedded in filename)
            checkpoints.sort(key=lambda x: float(str(x).split("val_total_loss=")[-1].split("-")[0]))
            checkpoint_path = str(checkpoints[0])
        else:  # "last"
            last_checkpoint = checkpoint_dir / "last.ckpt"
            if not last_checkpoint.exists():
                raise FileNotFoundError(f"Last checkpoint not found: {last_checkpoint}")
            checkpoint_path = str(last_checkpoint)
    
    # Load model from checkpoint
    model = FutureVoxModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    
    # Move to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model


def load_phoneme_dict(config):
    """
    Load phoneme dictionary
    
    Args:
        config: Configuration object
        
    Returns:
        Phoneme dictionary
    """
    phoneme_dict_path = config.preprocessing.text.phoneme_dict
    
    if not Path(phoneme_dict_path).exists():
        raise FileNotFoundError(f"Phoneme dictionary not found: {phoneme_dict_path}")
    
    with open(phoneme_dict_path, 'r') as f:
        phoneme_dict = json.load(f)
    
    return phoneme_dict


def synthesize(model, text, speaker_mixture, language_mixture, config, output_dir, output_name=None):
    """
    Synthesize speech from text
    
    Args:
        model: Trained FutureVox+ model
        text: Input text
        speaker_mixture: Speaker ID mixture
        language_mixture: Language ID mixture
        config: Configuration object
        output_dir: Directory to save outputs
        output_name: Base name for output files
        
    Returns:
        Path to generated audio file
    """
    device = next(model.parameters()).device
    
    # Process text to phonemes
    phonemes = text_to_phonemes(text, config.preprocessing.text.g2p_model)
    phoneme_ids = torch.tensor([model.phoneme_dict.get(p, 1) for p in phonemes], dtype=torch.long).to(device)
    
    # Add start and end tokens
    phoneme_ids = torch.cat([
        torch.tensor([2], device=device),  # BOS token
        phoneme_ids,
        torch.tensor([3], device=device)   # EOS token
    ])
    
    # Prepare speaker and language mixtures
    speaker_ids = torch.tensor(speaker_mixture, dtype=torch.float).to(device)
    language_ids = torch.tensor(language_mixture, dtype=torch.float).to(device)
    
    # Forward pass through model
    with torch.no_grad():
        # First pass: Generate durations and F0
        durations = model.forward_duration(phoneme_ids.unsqueeze(0), speaker_ids.unsqueeze(0), language_ids.unsqueeze(0))
        
        # Round durations to integers
        durations = torch.round(durations).squeeze(0)
        
        # Calculate total frames
        total_frames = int(durations.sum().item())
        
        # Generate F0 contour
        f0 = model.forward_f0(phoneme_ids.unsqueeze(0), durations.unsqueeze(0), 
                                speaker_ids.unsqueeze(0), language_ids.unsqueeze(0))
        
        # Second pass: Generate full output
        batch = {
            'phonemes': phoneme_ids.unsqueeze(0),
            'durations': durations.unsqueeze(0),
            'f0': f0,
            'speaker_ids': speaker_ids.unsqueeze(0),
            'language_ids': language_ids.unsqueeze(0)
        }
        
        output = model(batch)
        
        # Get output tensors
        mel_pred = output['mel_pred']
        waveform = output['waveform']
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output file names
    timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = output_name or f"futurevox_{timestamp}"
    
    audio_path = output_dir / f"{output_name}.wav"
    mel_path = output_dir / f"{output_name}_mel.png"
    f0_path = output_dir / f"{output_name}_f0.png"
    
    # Save audio
    save_audio(waveform.cpu(), audio_path, config.preprocessing.audio.sample_rate)
    
    # Save mel-spectrogram plot
    plt.figure(figsize=(10, 4))
    plot_mel_spectrogram(mel_pred.cpu().squeeze().transpose(0, 1).numpy())
    plt.tight_layout()
    plt.savefig(mel_path)
    plt.close()
    
    # Save F0 contour plot
    plt.figure(figsize=(10, 3))
    plot_f0_contour(f0.cpu().squeeze().numpy(), title="F0 Contour")
    plt.tight_layout()
    plt.savefig(f0_path)
    plt.close()
    
    return audio_path


def main():
    """
    Main function for inference
    """
    parser = argparse.ArgumentParser(description="FutureVox+ Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default="best", help="Path to model checkpoint or 'best'/'last'")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--text_file", type=str, default=None, help="File containing text to synthesize")
    parser.add_argument("--speaker_ids", type=str, default="0:0.8;1:0.2", help="Speaker ID mixture (id:weight;id:weight)")
    parser.add_argument("--language_ids", type=str, default="0:1.0", help="Language ID mixture (id:weight;id:weight)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set output directory
    output_dir = args.output_dir or config.inference.output_dir
    
    # Load model
    model = load_model(config, args.checkpoint or config.inference.checkpoint_path)
    
    # Parse speaker and language mixtures
    def parse_mixture(mixture_str):
        """Parse mixture string into list of [id, weight] pairs"""
        result = []
        for pair in mixture_str.split(';'):
            if not pair:
                continue
            id_str, weight_str = pair.split(':')
            result.append([int(id_str), float(weight_str)])
        return result
    
    speaker_mixture = parse_mixture(args.speaker_ids)
    language_mixture = parse_mixture(args.language_ids)
    
    # Get text to synthesize
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
    elif args.text:
        lines = [args.text]
    else:
        lines = ["Hello, I am FutureVox+, a multi-speaker multi-lingual voice synthesis model."]
    
    # Synthesize each line
    for i, text in enumerate(tqdm(lines, desc="Synthesizing")):
        output_name = f"output_{i+1}" if len(lines) > 1 else "output"
        audio_path = synthesize(
            model, 
            text, 
            speaker_mixture, 
            language_mixture, 
            config, 
            output_dir,
            output_name
        )
        print(f"Generated audio saved to {audio_path}")


if __name__ == "__main__":
    main()