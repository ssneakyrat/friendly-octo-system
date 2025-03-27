import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchaudio
import matplotlib.pyplot as plt
import io
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR, CosineAnnealingLR
from torch.optim import AdamW

from .modules.speaker_mixture import SpeakerMixtureModule
from .modules.language_mixture import LanguageMixtureModule
from .modules.f0_processor import F0ProcessorModule
from .modules.acoustic_gen import AcousticGenerator
from .modules.vocoder import Vocoder
from utils.audio import plot_mel_spectrogram


class FutureVoxModel(pl.LightningModule):
    """
    FutureVox+ main model class
    
    This class integrates all the components of the FutureVox+ model:
    - Speaker Mixture Module
    - Language Mixture Module
    - F0 Processing Module
    - Acoustic Feature Generator
    - Neural Vocoder
    
    The model handles multi-speaker, multi-language voice synthesis with
    mixing capabilities for both speaker characteristics and language accents.
    """
    
    def __init__(self, config):
        """
        Initialize the FutureVox+ model
        
        Args:
            config: Configuration object containing model hyperparameters
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize all model components
        self.speaker_mixture = SpeakerMixtureModule(
            config.model.speaker_mixture.embedding_dim,
            config.model.speaker_mixture.num_speakers,
            config.model.speaker_mixture.hidden_dims,
            config.model.speaker_mixture.attribute_net_dims
        )
        
        self.language_mixture = LanguageMixtureModule(
            config.model.language_mixture.embedding_dim,
            config.model.language_mixture.num_languages,
            config.model.language_mixture.num_phonemes,
            config.model.language_mixture.hidden_dims,
            config.model.language_mixture.accent_strength_dims
        )
        
        self.f0_processor = F0ProcessorModule(
            config.model.f0_processor.input_dim,
            config.model.f0_processor.hidden_dims,
            config.model.f0_processor.num_registers,
            config.model.f0_processor.decomp_dims
        )
        
        self.acoustic_generator = AcousticGenerator(
            config.model.acoustic_generator.model_dim,
            config.model.acoustic_generator.num_layers,
            config.model.acoustic_generator.num_heads,
            config.model.acoustic_generator.ffn_dim,
            config.model.acoustic_generator.dropout,
            config.model.acoustic_generator.cross_attention_layers
        )
        
        self.vocoder = Vocoder(
            config.model.vocoder.input_dim,
            config.model.vocoder.hidden_dims,
            config.model.vocoder.upsample_rates,
            config.model.vocoder.upsample_kernel_sizes,
            config.model.vocoder.resblock_kernel_sizes,
            config.model.vocoder.resblock_dilation_sizes,
            config.model.vocoder.gan_feat_matching_dim,
            config.model.vocoder.multi_period_discriminator_periods
        )
        
        # Loss functions
        self.mel_loss = nn.L1Loss()
        self.feature_loss = nn.MSELoss()
        self.f0_loss = nn.L1Loss()
        self.duration_loss = nn.MSELoss()
        
    def forward(self, batch):
        """
        Forward pass through the FutureVox+ model
        
        Args:
            batch: Dictionary containing:
                - phonemes: Tokenized G2P phonemes [batch_size, seq_len]
                - durations: Phoneme durations [batch_size, seq_len]
                - f0: Fundamental frequency contours [batch_size, time_steps, 1]
                - mel_target: Target mel-spectrograms [batch_size, n_mels, time_steps]
                - speaker_ids: Speaker ID mixtures [batch_size, n_speakers, 2]
                - language_ids: Language ID mixtures [batch_size, n_languages, 2]
                
        Returns:
            Dictionary containing:
                - mel_pred: Predicted mel-spectrograms
                - waveform: Generated audio waveforms
                - processed_f0: Processed F0 contours
                - predicted_durations: Predicted phoneme durations (if applicable)
        """
        phonemes = batch['phonemes']
        durations = batch['durations']
        f0 = batch['f0']
        mel_target = batch.get('mel_spectrogram')
        speaker_ids = batch['speaker_ids']
        language_ids = batch['language_ids']
        
        # Process speaker information to get speaker embeddings
        speaker_emb = self.speaker_mixture(speaker_ids)
        
        # Process language information to get language and phoneme embeddings
        language_emb, phoneme_emb = self.language_mixture(language_ids, phonemes)
        
        # Process F0 with respect to speaker and language characteristics
        processed_f0 = self.f0_processor(f0, speaker_emb, language_emb)
        
        # Generate acoustic features (mel-spectrogram)
        predicted_mel = self.acoustic_generator(
            phoneme_emb, durations, processed_f0, speaker_emb, language_emb
        )
        
        # Generate waveform from predicted mel-spectrogram
        waveform = self.vocoder(predicted_mel)
        
        return {
            'mel_pred': predicted_mel,
            'waveform': waveform,
            'processed_f0': processed_f0
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step for FutureVox+
        
        Args:
            batch: Batch data
            batch_idx: Index of the current batch
            
        Returns:
            Total loss for the batch
        """
        outputs = self(batch)
        mel_target = batch['mel_spectrogram']
        f0_target = batch['f0']
        
        # Calculate losses
        mel_loss = self.mel_loss(outputs['mel_pred'], mel_target)
        f0_loss = self.f0_loss(outputs['processed_f0'], f0_target)
        
        # Weight the losses according to config
        weighted_mel_loss = self.config.training.loss_weights.mel_loss * mel_loss
        weighted_f0_loss = self.config.training.loss_weights.f0_loss * f0_loss
        
        # Compute total loss
        total_loss = weighted_mel_loss + weighted_f0_loss
        
        # Log metrics
        self.log('train/mel_loss', mel_loss, prog_bar=True)
        self.log('train/f0_loss', f0_loss)
        self.log('train/total_loss', total_loss)
        
        # Log audio samples occasionally
        log_freq = self.config.logger.audio.log_every_n_steps
        if batch_idx % log_freq == 0:
            self._log_audio_samples(outputs['waveform'], mel_target, outputs['mel_pred'], 'train')
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for FutureVox+
        
        Args:
            batch: Batch data
            batch_idx: Index of the current batch
            
        Returns:
            Total validation loss
        """
        outputs = self(batch)
        mel_target = batch['mel_spectrogram']
        f0_target = batch['f0']
        
        # Calculate losses
        mel_loss = self.mel_loss(outputs['mel_pred'], mel_target)
        f0_loss = self.f0_loss(outputs['processed_f0'], f0_target)
        
        # Weight the losses according to config
        weighted_mel_loss = self.config.training.loss_weights.mel_loss * mel_loss
        weighted_f0_loss = self.config.training.loss_weights.f0_loss * f0_loss
        
        # Compute total loss
        total_loss = weighted_mel_loss + weighted_f0_loss
        
        # Log metrics
        self.log('val/mel_loss', mel_loss, prog_bar=True)
        self.log('val/f0_loss', f0_loss)
        self.log('val/total_loss', total_loss)
        
        # Log audio samples from validation
        if batch_idx < self.config.validation.num_audio_samples:
            self._log_audio_samples(outputs['waveform'], mel_target, outputs['mel_pred'], 'val')
        
        return total_loss
    
    def _log_audio_samples(self, waveform, mel_target, mel_pred, stage='train'):
        """
        Log audio samples and spectrograms to TensorBoard
        
        Args:
            waveform: Generated audio waveforms
            mel_target: Target mel-spectrograms
            mel_pred: Predicted mel-spectrograms
            stage: 'train' or 'val'
        """
        # Take a subset of samples to log
        n_samples = min(self.config.logger.audio.n_samples_per_batch, waveform.size(0))
        
        for i in range(n_samples):
            # Log generated audio
            sample_rate = self.config.logger.audio.sample_rate
            audio_data = waveform[i].detach().cpu()
            
            # Normalize audio for logging
            audio_data = audio_data / torch.max(torch.abs(audio_data))
            
            self.logger.experiment.add_audio(
                f'{stage}/generated_audio_{i}',
                audio_data,
                self.global_step,
                sample_rate
            )
            
            # Create and log spectrogram comparison
            fig = plt.figure(figsize=(10, 6))
            
            # Plot target mel
            plt.subplot(2, 1, 1)
            plt.title("Ground Truth Mel-Spectrogram")
            plot_mel_spectrogram(mel_target[i].detach().cpu().numpy())
            
            # Plot predicted mel
            plt.subplot(2, 1, 2)
            plt.title("Predicted Mel-Spectrogram")
            plot_mel_spectrogram(mel_pred[i].detach().cpu().numpy())
            
            plt.tight_layout()
            
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            
            # Log image to TensorBoard
            img = plt.imread(buf)
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            self.logger.experiment.add_image(f'{stage}/mel_comparison_{i}', img, self.global_step)
    
    def configure_optimizers(self):
        """
        Configure optimizers for training
        
        Returns:
            Optimizer and learning rate scheduler
        """
        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Create scheduler based on config
        scheduler_type = self.config.training.scheduler
        scheduler_params = self.config.training.scheduler_params
        
        if scheduler_type == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=scheduler_params.pct_start
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}
            
        elif scheduler_type == "ExponentialLR":
            scheduler = ExponentialLR(
                optimizer,
                gamma=scheduler_params.gamma
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
            
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_params.T_max
            )
            scheduler_config = {"scheduler": scheduler, "interval": "epoch"}
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}