import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Qt
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import librosa.display


def plot_mel_spectrogram(mel_spec, title='Mel-Spectrogram', ax=None):
    """
    Plot mel-spectrogram
    
    Args:
        mel_spec: Mel-spectrogram array of shape [n_mels, time] or [time, n_mels]
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.detach().cpu().numpy()
    
    # Ensure correct orientation (n_mels should be first dimension)
    if mel_spec.shape[0] < mel_spec.shape[1]:
        mel_spec = mel_spec.T
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot mel-spectrogram
    img = librosa.display.specshow(
        mel_spec,
        y_axis='mel',
        x_axis='time',
        ax=ax
    )
    
    # Add colorbar
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    
    # Set title
    ax.set_title(title)
    
    return ax


def plot_waveform(waveform, sample_rate=22050, title='Waveform', ax=None):
    """
    Plot audio waveform
    
    Args:
        waveform: Audio waveform array
        sample_rate: Audio sample rate
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    # Flatten if needed
    if waveform.ndim > 1:
        waveform = waveform.flatten()
    
    # Create time axis
    time = np.arange(0, len(waveform)) / sample_rate
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot waveform
    ax.plot(time, waveform)
    
    # Set labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True)
    
    return ax


def plot_f0_contour(f0, title='F0 Contour', ax=None):
    """
    Plot F0 contour
    
    Args:
        f0: F0 contour array
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(f0, torch.Tensor):
        f0 = f0.detach().cpu().numpy()
    
    # Flatten if needed
    if f0.ndim > 1:
        f0 = f0.flatten()
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot F0 contour
    ax.plot(f0)
    
    # Set labels
    ax.set_xlabel('Frame')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True)
    
    return ax


def plot_attention(attention, title='Attention', ax=None):
    """
    Plot attention matrix
    
    Args:
        attention: Attention matrix of shape [target_len, source_len]
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot attention matrix
    img = ax.imshow(attention, aspect='auto', origin='lower', cmap='hot')
    
    # Add colorbar
    plt.colorbar(img, ax=ax)
    
    # Set labels
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title(title)
    
    return ax


def plot_alignment(durations, title='Alignment', ax=None):
    """
    Plot alignment based on durations
    
    Args:
        durations: Durations array
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(durations, torch.Tensor):
        durations = durations.detach().cpu().numpy()
    
    # Flatten if needed
    if durations.ndim > 1:
        durations = durations.flatten()
    
    # Create alignment matrix
    total_frames = int(np.sum(durations))
    alignment = np.zeros((total_frames, len(durations)))
    
    frame_idx = 0
    for i, duration in enumerate(durations):
        for j in range(int(duration)):
            if frame_idx < total_frames:
                alignment[frame_idx, i] = 1.0
                frame_idx += 1
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot alignment matrix
    img = ax.imshow(alignment, aspect='auto', origin='lower', cmap='Blues')
    
    # Add colorbar
    plt.colorbar(img, ax=ax)
    
    # Set labels
    ax.set_xlabel('Phoneme')
    ax.set_ylabel('Frame')
    ax.set_title(title)
    
    return ax


def plot_speaker_mixture(speaker_weights, speaker_names=None, title='Speaker Mixture', ax=None):
    """
    Plot speaker mixture weights
    
    Args:
        speaker_weights: Speaker weights array
        speaker_names: List of speaker names (optional)
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(speaker_weights, torch.Tensor):
        speaker_weights = speaker_weights.detach().cpu().numpy()
    
    # Flatten if needed
    if speaker_weights.ndim > 1:
        speaker_weights = speaker_weights.flatten()
    
    # Create labels
    if speaker_names is None:
        speaker_names = [f"Speaker {i+1}" for i in range(len(speaker_weights))]
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot bar chart
    ax.bar(speaker_names, speaker_weights)
    
    # Set labels
    ax.set_xlabel('Speaker')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    
    # Add values above bars
    for i, v in enumerate(speaker_weights):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    return ax


def plot_language_mixture(language_weights, language_names=None, title='Language Mixture', ax=None):
    """
    Plot language mixture weights
    
    Args:
        language_weights: Language weights array
        language_names: List of language names (optional)
        title: Plot title
        ax: Matplotlib axis (optional)
        
    Returns:
        Matplotlib axis
    """
    # Convert to numpy
    if isinstance(language_weights, torch.Tensor):
        language_weights = language_weights.detach().cpu().numpy()
    
    # Flatten if needed
    if language_weights.ndim > 1:
        language_weights = language_weights.flatten()
    
    # Create labels
    if language_names is None:
        language_names = [f"Language {i+1}" for i in range(len(language_weights))]
    
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot bar chart
    ax.bar(language_names, language_weights)
    
    # Set labels
    ax.set_xlabel('Language')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    
    # Add values above bars
    for i, v in enumerate(language_weights):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    return ax


def create_generation_figure(mel_spec, f0, waveform=None, attention=None, speaker_weights=None, language_weights=None):
    """
    Create a comprehensive figure for speech generation
    
    Args:
        mel_spec: Mel-spectrogram
        f0: F0 contour
        waveform: Audio waveform (optional)
        attention: Attention matrix (optional)
        speaker_weights: Speaker weights (optional)
        language_weights: Language weights (optional)
        
    Returns:
        Matplotlib figure
    """
    # Determine number of subplots
    n_plots = 2  # Mel-spectrogram and F0 are always shown
    if waveform is not None:
        n_plots += 1
    if attention is not None:
        n_plots += 1
    
    # Create figure
    fig = plt.figure(figsize=(12, n_plots * 3))
    
    # Determine layout
    if n_plots <= 2:
        n_rows, n_cols = n_plots, 1
    else:
        n_rows, n_cols = (n_plots + 1) // 2, 2
    
    # Plot mel-spectrogram
    ax1 = plt.subplot(n_rows, n_cols, 1)
    plot_mel_spectrogram(mel_spec, title='Generated Mel-Spectrogram', ax=ax1)
    
    # Plot F0 contour
    ax2 = plt.subplot(n_rows, n_cols, 2)
    plot_f0_contour(f0, title='F0 Contour', ax=ax2)
    
    plot_idx = 3
    
    # Plot waveform if provided
    if waveform is not None:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_waveform(waveform, title='Generated Waveform', ax=ax)
        plot_idx += 1
    
    # Plot attention if provided
    if attention is not None:
        ax = plt.subplot(n_rows, n_cols, plot_idx)
        plot_attention(attention, title='Attention', ax=ax)
        plot_idx += 1
    
    # Add speaker and language mixture insets if provided
    if speaker_weights is not None or language_weights is not None:
        # Create a smaller figure for the mixtures
        mixture_fig, mixture_axes = plt.subplots(
            1, 
            2 if speaker_weights is not None and language_weights is not None else 1, 
            figsize=(8, 3)
        )
        
        # Plot speaker mixture
        if speaker_weights is not None:
            if isinstance(mixture_axes, np.ndarray):
                ax_spk = mixture_axes[0]
            else:
                ax_spk = mixture_axes
            plot_speaker_mixture(speaker_weights, title='Speaker Mixture', ax=ax_spk)
        
        # Plot language mixture
        if language_weights is not None:
            if isinstance(mixture_axes, np.ndarray) and speaker_weights is not None:
                ax_lang = mixture_axes[1]
            else:
                ax_lang = mixture_axes
            plot_language_mixture(language_weights, title='Language Mixture', ax=ax_lang)
        
        plt.tight_layout()
    
    # Adjust main figure layout
    plt.tight_layout()
    
    return fig