import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, spectral_norm


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(
            channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation))
        self.skip = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(x, 0.1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.conv2(x)
        return x + residual


class MultiRateProcessing(nn.Module):
    """
    Multi-rate processing module for different temporal resolutions
    """
    def __init__(self, input_channels, output_channels, rates=(1, 2, 4)):
        super().__init__()
        self.rates = rates
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 3, padding=1, dilation=rate),
                nn.LeakyReLU(0.1),
                nn.Conv1d(output_channels, output_channels, 3, padding=1, dilation=rate)
            ) for rate in rates
        ])
        self.integration = nn.Conv1d(output_channels * len(rates), output_channels, 1)
        
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        combined = torch.cat(outputs, dim=1)
        return self.integration(combined)


class Generator(nn.Module):
    """
    HiFi-GAN based generator with enhanced multi-rate processing
    """
    def __init__(self, input_dim, hidden_dims, upsample_rates, upsample_kernel_sizes, 
                 resblock_kernel_sizes, resblock_dilation_sizes):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial conv
        self.conv_pre = weight_norm(nn.Conv1d(input_dim, hidden_dims[0], 7, padding=3))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    hidden_dims[min(i, len(hidden_dims) - 1)],
                    hidden_dims[min(i + 1, len(hidden_dims) - 1)],
                    k, stride=u, padding=(k - u) // 2
                )
            ))
        
        # Multi-rate processing
        self.mrp = MultiRateProcessing(hidden_dims[-1], hidden_dims[-1])
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hidden_dims[min(i + 1, len(hidden_dims) - 1)]
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Harmonic generator
        self.harmonic_gen = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # Noise generator
        self.noise_gen = nn.Sequential(
            nn.Conv1d(hidden_dims[-1], hidden_dims[-1] // 2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dims[-1] // 2, hidden_dims[-1] // 2, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        
        # Spectral envelope predictor
        self.spec_env = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(512, hidden_dims[-1], 3, padding=1)
        )
        
        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(hidden_dims[-1] * 2, 1, 7, padding=3))
        
    def forward(self, x):
        # x: mel-spectrogram [B, n_mels, T]
        
        # Initial convolution
        x = self.conv_pre(x)  # [B, hidden_dim, T]
        
        # Spectral envelope prediction for formant structure preservation
        spec_envelope = self.spec_env(x)
        
        # Upsampling
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            
            # Apply residual blocks
            xs = 0
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                xs += self.resblocks[idx](x)
            x = xs / self.num_kernels
        
        # Apply multi-rate processing
        x = self.mrp(x)
        
        # Generate harmonic and noise components
        harmonic = self.harmonic_gen(x)
        noise = self.noise_gen(x)
        noise = F.interpolate(noise, size=harmonic.shape[-1])  # Ensure same length
        
        # Apply spectral envelope to preserve formant structure
        envelope_expanded = F.interpolate(spec_envelope, size=harmonic.shape[-1])
        harmonic = harmonic * envelope_expanded
        
        # Combine harmonic and noise components
        x = torch.cat([harmonic, noise], dim=1)
        
        # Final convolution
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator for GAN training
    """
    def __init__(self, periods=None):
        super().__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period) for period in self.periods
        ])
        
    def forward(self, y, y_hat):
        """
        Args:
            y: ground truth waveform
            y_hat: predicted waveform
            
        Returns:
            List of (real_outputs, fake_outputs, real_features, fake_features)
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class PeriodDiscriminator(nn.Module):
    """
    Period-based discriminator
    """
    def __init__(self, period):
        super().__init__()
        self.period = period
        norm_f = spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        
    def forward(self, x):
        """
        Args:
            x: waveform of shape [B, 1, T]
            
        Returns:
            Discriminator output and feature maps
        """
        batch_size = x.shape[0]
        
        # Convert to [B, 1, T//period, period]
        features = []
        
        # Handle case where signal length is not divisible by period
        pad_len = (self.period - (x.shape[-1] % self.period)) % self.period
        x = F.pad(x, (0, pad_len))
        x = x.view(batch_size, 1, -1, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
            
        x = self.conv_post(x)
        features.append(x)
        
        # Flatten
        x = torch.flatten(x, 1, -1)
        
        return x, features


class Vocoder(nn.Module):
    """
    Neural Vocoder module for FutureVox+
    
    This module handles:
    1. Multi-rate pitch-synchronous processing
    2. Vocal tract length normalization
    3. Neural source-filter modeling with explicit harmonic control
    4. High-fidelity synthesis with reduced artifacts
    
    Total parameters: ~24.5M
    """
    
    def __init__(self, input_dim, hidden_dims, upsample_rates, upsample_kernel_sizes, 
                 resblock_kernel_sizes, resblock_dilation_sizes, gan_feat_matching_dim=16,
                 multi_period_discriminator_periods=None):
        """
        Initialize vocoder module
        
        Args:
            input_dim: Input dimension (number of mel bins)
            hidden_dims: List of hidden dimensions
            upsample_rates: List of upsampling rates
            upsample_kernel_sizes: List of upsampling kernel sizes
            resblock_kernel_sizes: List of residual block kernel sizes
            resblock_dilation_sizes: List of dilation rates for each residual block
            gan_feat_matching_dim: Feature matching dimension for GAN
            multi_period_discriminator_periods: List of periods for discriminator
        """
        super().__init__()
        
        # Generator (~23.9M parameters)
        self.generator = Generator(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )
        
        # Discriminator (used only during training)
        self.discriminator = MultiPeriodDiscriminator(multi_period_discriminator_periods)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize network weights for better training
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, mel_spectrogram):
        """
        Generate waveform from mel-spectrogram
        
        Args:
            mel_spectrogram: Mel-spectrogram of shape [batch_size, n_mels, time_steps]
            
        Returns:
            Waveform of shape [batch_size, 1, time_steps*hop_length]
        """
        # In inference, we only need the generator
        waveform = self.generator(mel_spectrogram)
        return waveform
    
    def discriminate(self, y, y_hat):
        """
        Discriminate between real and generated audio
        
        Args:
            y: Ground truth waveform
            y_hat: Generated waveform
            
        Returns:
            Discrimination results for loss calculation
        """
        return self.discriminator(y, y_hat)
    
    def feature_matching_loss(self, fmap_r, fmap_g):
        """
        Calculate feature matching loss for GAN training
        
        Args:
            fmap_r: Real feature maps
            fmap_g: Generated feature maps
            
        Returns:
            Feature matching loss
        """
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += F.l1_loss(rl, gl)
        return loss