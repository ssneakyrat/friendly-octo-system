import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require Qt
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn


def load_audio(file_path, sample_rate=22050):
    """
    Load audio file and resample if necessary
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tensor containing audio waveform
    """
    waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != sample_rate:
        waveform = F.resample(waveform, sr, sample_rate)
    
    return waveform


def normalize_audio(waveform):
    """
    Normalize audio waveform to range [-1, 1]
    
    Args:
        waveform: Audio waveform tensor
        
    Returns:
        Normalized waveform
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def preemphasis(waveform, coef=0.97):
    """
    Apply pre-emphasis to waveform
    
    Args:
        waveform: Audio waveform tensor
        coef: Pre-emphasis coefficient
        
    Returns:
        Pre-emphasized waveform
    """
    return torch.cat([waveform[:, :1], waveform[:, 1:] - coef * waveform[:, :-1]], dim=1)


def deemphasis(waveform, coef=0.97):
    """
    Apply de-emphasis to waveform
    
    Args:
        waveform: Audio waveform tensor
        coef: De-emphasis coefficient
        
    Returns:
        De-emphasized waveform
    """
    result = torch.zeros_like(waveform)
    result[:, 0] = waveform[:, 0]
    for i in range(1, waveform.size(1)):
        result[:, i] = waveform[:, i] + coef * result[:, i-1]
    return result


def compute_mel_spectrogram(
    waveform, 
    sample_rate=22050, 
    n_fft=1024, 
    hop_length=256, 
    win_length=1024,
    n_mels=80, 
    f_min=0, 
    f_max=8000,
    power=1.0,
    normalized=True
):
    """
    Compute mel-spectrogram from waveform
    
    Args:
        waveform: Audio waveform tensor of shape [batch_size, samples]
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window size
        n_mels: Number of mel bands
        f_min: Minimum frequency
        f_max: Maximum frequency
        power: Power for spectrogram. 1.0 for amplitude, 2.0 for power
        normalized: Whether to normalize mel-spectrogram
        
    Returns:
        Mel-spectrogram of shape [batch_size, n_mels, time]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Ensure waveform has batch dimension
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Apply pre-emphasis
    waveform = preemphasis(waveform)
    
    # Create mel transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=power
    )
    
    # Compute mel-spectrogram
    mel_spec = mel_transform(waveform)
    
    # Apply log scaling
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    if normalized:
        mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    
    return mel_spec


def griffin_lim(mag_spec, n_fft=1024, hop_length=256, win_length=1024, n_iter=60):
    """
    Compute waveform from magnitude spectrogram using Griffin-Lim algorithm
    
    Args:
        mag_spec: Magnitude spectrogram of shape [batch_size, n_fft//2 + 1, time]
        n_fft: FFT size
        hop_length: Hop size
        win_length: Window size
        n_iter: Number of iterations
        
    Returns:
        Reconstructed waveform
    """
    batch_size = mag_spec.shape[0]
    device = mag_spec.device
    
    # Initialize random phase
    phase = torch.randn(mag_spec.shape, device=device) * 2 * np.pi
    
    # Complex spectrogram
    complex_spec = mag_spec * torch.exp(1j * phase)
    
    # Iterate
    for _ in range(n_iter):
        # Inverse STFT
        waveform = torch.istft(
            complex_spec, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length,
            window=torch.hann_window(win_length, device=device)
        )
        
        # STFT
        complex_spec_new = torch.stft(
            waveform, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length,
            window=torch.hann_window(win_length, device=device),
            return_complex=True
        )
        
        # Update phase
        phase = torch.angle(complex_spec_new)
        complex_spec = mag_spec * torch.exp(1j * phase)
    
    # Final inverse STFT
    waveform = torch.istft(
        complex_spec, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=torch.hann_window(win_length, device=device)
    )
    
    # Apply de-emphasis
    waveform = deemphasis(waveform)
    
    return waveform


def extract_f0(waveform, sample_rate=22050, hop_length=256, f0_min=65, f0_max=1000):
    """
    Extract F0 from waveform using PyWorld
    
    Args:
        waveform: Audio waveform tensor
        sample_rate: Sample rate
        hop_length: Hop size
        f0_min: Minimum F0
        f0_max: Maximum F0
        
    Returns:
        F0 contour of shape [batch_size, time]
    """
    # Import pyworld here to avoid making it a mandatory dependency
    try:
        import pyworld as pw
    except ImportError:
        raise ImportError("Please install PyWorld: pip install pyworld")
    
    # Convert to numpy for PyWorld
    wav_numpy = waveform.squeeze().cpu().numpy()
    
    # Extract F0
    f0, t = pw.dio(
        wav_numpy.astype(np.float64), 
        sample_rate, 
        frame_period=hop_length * 1000.0 / sample_rate,
        f0_floor=f0_min,
        f0_ceil=f0_max
    )
    
    # Refine F0
    f0 = pw.stonemask(wav_numpy.astype(np.float64), f0, t, sample_rate)
    
    # Convert back to tensor
    f0 = torch.from_numpy(f0).float().unsqueeze(0)
    
    # Replace zero values with a small positive value
    f0 = torch.where(f0 > 0, f0, torch.ones_like(f0) * 1e-3)
    
    return f0


def plot_mel_spectrogram(mel_spec, title="Mel-Spectrogram"):
    """
    Plot mel-spectrogram
    
    Args:
        mel_spec: Mel-spectrogram array of shape [n_mels, time]
        title: Plot title
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.cpu().numpy()
    
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()


def save_audio(waveform, file_path, sample_rate=22050):
    """
    Save audio waveform to file
    
    Args:
        waveform: Audio waveform tensor
        file_path: Path to save audio file
        sample_rate: Sample rate
    """
    if waveform.dim() > 2:
        waveform = waveform.squeeze(1)
    
    # Normalize if needed
    if torch.max(torch.abs(waveform)) > 1.0:
        waveform = normalize_audio(waveform)
    
    torchaudio.save(file_path, waveform.cpu(), sample_rate)