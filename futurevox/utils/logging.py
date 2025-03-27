import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display
from PIL import Image


def plot_to_tensor(fig):
    """
    Convert matplotlib figure to PyTorch tensor
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        PyTorch tensor of shape [3, height, width]
    """
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Load image from buffer
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Convert to PyTorch tensor
    # img_array shape: [height, width, channels]
    # tensor shape: [channels, height, width]
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    
    return tensor


def log_audio_to_tensorboard(logger, audio, sample_rate, tag, global_step):
    """
    Log audio to TensorBoard
    
    Args:
        logger: TensorBoard logger
        audio: Audio tensor of shape [batch_size, samples]
        sample_rate: Audio sample rate
        tag: Log tag
        global_step: Global step
    """
    # Normalize audio
    audio = audio / torch.max(torch.abs(audio))
    
    # Log audio
    logger.add_audio(tag, audio, global_step, sample_rate)


def log_mel_spectrogram_to_tensorboard(logger, mel_spec, tag, global_step):
    """
    Log mel-spectrogram to TensorBoard
    
    Args:
        logger: TensorBoard logger
        mel_spec: Mel-spectrogram tensor of shape [batch_size, n_mels, time]
        tag: Log tag
        global_step: Global step
    """
    # Convert to numpy
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot mel-spectrogram
    img = librosa.display.specshow(
        mel_spec[0] if mel_spec.ndim > 2 else mel_spec, 
        y_axis='mel', 
        x_axis='time', 
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(tag)
    plt.tight_layout()
    
    # Convert to tensor
    tensor = plot_to_tensor(fig)
    
    # Log to TensorBoard
    logger.add_image(tag, tensor, global_step)
    
    # Close figure
    plt.close(fig)


def log_f0_to_tensorboard(logger, f0, tag, global_step):
    """
    Log F0 contour to TensorBoard
    
    Args:
        logger: TensorBoard logger
        f0: F0 tensor of shape [batch_size, time, 1]
        tag: Log tag
        global_step: Global step
    """
    # Convert to numpy
    if isinstance(f0, torch.Tensor):
        f0 = f0.detach().cpu().numpy()
    
    # Reshape if needed
    if f0.ndim == 3:
        f0 = f0[0]  # Take first batch element
    if f0.ndim == 2 and f0.shape[1] == 1:
        f0 = f0.squeeze(1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Plot F0 contour
    ax.plot(f0)
    ax.set_title(f"{tag} - F0 Contour")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (frames)")
    ax.grid(True)
    plt.tight_layout()
    
    # Convert to tensor
    tensor = plot_to_tensor(fig)
    
    # Log to TensorBoard
    logger.add_image(f"{tag}/f0", tensor, global_step)
    
    # Close figure
    plt.close(fig)


def log_attention_to_tensorboard(logger, attention, tag, global_step):
    """
    Log attention weights to TensorBoard
    
    Args:
        logger: TensorBoard logger
        attention: Attention tensor of shape [batch_size, target_len, source_len]
        tag: Log tag
        global_step: Global step
    """
    # Convert to numpy
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()
    
    # Take first batch element
    if attention.ndim == 3:
        attention = attention[0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list(
        "attention_cmap", 
        [(0, 0, 0, 0), (1, 0, 0, 1)],  # Transparent to red
        N=256
    )
    
    # Plot attention
    img = ax.imshow(attention, aspect='auto', origin='lower', cmap=cmap)
    fig.colorbar(img, ax=ax)
    ax.set_title(f"{tag} - Attention")
    ax.set_ylabel("Target")
    ax.set_xlabel("Source")
    plt.tight_layout()
    
    # Convert to tensor
    tensor = plot_to_tensor(fig)
    
    # Log to TensorBoard
    logger.add_image(f"{tag}/attention", tensor, global_step)
    
    # Close figure
    plt.close(fig)


def log_mel_comparison_to_tensorboard(logger, mel_target, mel_pred, tag, global_step):
    """
    Log comparison of target and predicted mel-spectrograms to TensorBoard
    
    Args:
        logger: TensorBoard logger
        mel_target: Target mel-spectrogram tensor of shape [batch_size, n_mels, time]
        mel_pred: Predicted mel-spectrogram tensor of shape [batch_size, n_mels, time]
        tag: Log tag
        global_step: Global step
    """
    # Convert to numpy
    if isinstance(mel_target, torch.Tensor):
        mel_target = mel_target.detach().cpu().numpy()
    if isinstance(mel_pred, torch.Tensor):
        mel_pred = mel_pred.detach().cpu().numpy()
    
    # Take first batch element
    if mel_target.ndim == 3:
        mel_target = mel_target[0]
    if mel_pred.ndim == 3:
        mel_pred = mel_pred[0]
    
    # Transpose if needed
    if mel_target.shape[0] < mel_target.shape[1]:
        mel_target = mel_target.T
    if mel_pred.shape[0] < mel_pred.shape[1]:
        mel_pred = mel_pred.T
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot target mel-spectrogram
    librosa.display.specshow(
        mel_target, 
        y_axis='mel', 
        x_axis='time', 
        ax=axes[0]
    )
    axes[0].set_title(f"{tag} - Target Mel-Spectrogram")
    
    # Plot predicted mel-spectrogram
    img = librosa.display.specshow(
        mel_pred, 
        y_axis='mel', 
        x_axis='time', 
        ax=axes[1]
    )
    axes[1].set_title(f"{tag} - Predicted Mel-Spectrogram")
    
    # Add colorbar
    fig.colorbar(img, ax=axes, format="%+2.0f dB")
    
    plt.tight_layout()
    
    # Convert to tensor
    tensor = plot_to_tensor(fig)
    
    # Log to TensorBoard
    logger.add_image(f"{tag}/mel_comparison", tensor, global_step)
    
    # Close figure
    plt.close(fig)


def log_weights_and_biases(logger, model, global_step):
    """
    Log model weights and biases to TensorBoard
    
    Args:
        logger: TensorBoard logger
        model: PyTorch model
        global_step: Global step
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.add_histogram(f"parameters/{name}", param.data, global_step)
            if param.grad is not None:
                logger.add_histogram(f"gradients/{name}", param.grad.data, global_step)


def log_learning_rate(logger, scheduler, global_step):
    """
    Log learning rate to TensorBoard
    
    Args:
        logger: TensorBoard logger
        scheduler: Learning rate scheduler
        global_step: Global step
    """
    if scheduler is None:
        return
    
    # Get current learning rate
    if hasattr(scheduler, "get_last_lr"):
        lr = scheduler.get_last_lr()[0]
    else:
        lr = scheduler.optimizer.param_groups[0]["lr"]
    
    # Log learning rate
    logger.add_scalar("training/learning_rate", lr, global_step)


def log_prediction_examples(logger, batch, outputs, global_step, num_examples=2):
    """
    Log examples of model predictions to TensorBoard
    
    Args:
        logger: TensorBoard logger
        batch: Batch of input data
        outputs: Model outputs
        global_step: Global step
        num_examples: Number of examples to log
    """
    # Limit number of examples
    n = min(num_examples, batch['mel_spectrogram'].size(0))
    
    for i in range(n):
        # Extract tensors
        mel_target = batch['mel_spectrogram'][i]
        mel_pred = outputs['mel_pred'][i]
        waveform = outputs['waveform'][i] if 'waveform' in outputs else None
        f0 = outputs['processed_f0'][i] if 'processed_f0' in outputs else None
        attention = outputs['attention'][i] if 'attention' in outputs else None
        
        # Log mel-spectrogram comparison
        log_mel_comparison_to_tensorboard(
            logger, 
            mel_target.unsqueeze(0), 
            mel_pred.unsqueeze(0), 
            f"example_{i+1}", 
            global_step
        )
        
        # Log audio if available
        if waveform is not None:
            log_audio_to_tensorboard(
                logger, 
                waveform.unsqueeze(0), 
                22050,  # Sample rate
                f"example_{i+1}/audio", 
                global_step
            )
        
        # Log F0 if available
        if f0 is not None:
            log_f0_to_tensorboard(
                logger, 
                f0.unsqueeze(0), 
                f"example_{i+1}", 
                global_step
            )
        
        # Log attention if available
        if attention is not None:
            log_attention_to_tensorboard(
                logger, 
                attention.unsqueeze(0), 
                f"example_{i+1}", 
                global_step
            )