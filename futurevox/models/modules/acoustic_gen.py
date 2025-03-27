import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..layers.transformer import TransformerEncoder, TransformerCrossAttention


class AcousticGenerator(nn.Module):
    """
    Acoustic Feature Generation Module for FutureVox+
    
    This module handles:
    1. Transformer-based sequence modeling for mel-spectrogram prediction
    2. Attention mechanisms for aligning phonetic and acoustic features
    3. Formant structure preservation across speaker/language mixtures
    4. Joint optimization of all conditioning factors
    
    Total parameters: ~8.9M
    """
    
    def __init__(self, model_dim, num_layers, num_heads, ffn_dim, dropout=0.1, cross_attention_layers=2):
        """
        Initialize acoustic generator module
        
        Args:
            model_dim: Hidden dimension of the transformer model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            ffn_dim: Feed-forward network dimension
            dropout: Dropout rate
            cross_attention_layers: Number of cross-attention layers
        """
        super().__init__()
        self.model_dim = model_dim
        
        # Input embedding (~0.15M parameters)
        self.input_embedding = nn.Linear(model_dim, model_dim)
        
        # Transformer encoder (~7.12M parameters)
        self.encoder = TransformerEncoder(
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        # Cross-attention layers (~0.74M parameters)
        self.cross_attention_layers = nn.ModuleList([
            TransformerCrossAttention(model_dim, num_heads, dropout)
            for _ in range(cross_attention_layers)
        ])
        
        # Formant tracking network (~0.13M parameters)
        self.formant_tracker = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64)
        )
        
        # Formant preservation network (~0.23M parameters)
        self.formant_preserver = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(model_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128)
        )
        
        # Condition adapters (~0.78M parameters)
        # Speaker adaptation
        self.speaker_adapter = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(model_dim, 256)
        )
        
        # Language adaptation
        self.language_adapter = nn.Sequential(
            nn.Linear(512, model_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(model_dim, 256)
        )
        
        # F0 adaptation
        self.f0_adapter = nn.Sequential(
            nn.Linear(256, model_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(model_dim, 256)
        )
        
        # Duration adapter
        self.duration_adapter = nn.Linear(1, model_dim)
        
        # Output projection - projects to mel-spectrogram
        self.output_proj = nn.Linear(model_dim, 80)  # 80 mel bins
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize network weights for better training
        """
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier for most linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _create_alignment(self, durations):
        """
        Create alignment matrix from phoneme durations
        
        Args:
            durations: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Alignment matrix of shape [batch_size, mel_len, seq_len]
        """
        batch_size, seq_len = durations.shape
        mel_len = torch.sum(durations, dim=1).long()
        max_mel_len = torch.max(mel_len).item()
        
        # Create alignment matrices
        alignments = []
        
        for i in range(batch_size):
            alignment = torch.zeros(max_mel_len, seq_len, device=durations.device)
            
            # Fill in alignment based on durations
            pos = 0
            for j in range(seq_len):
                dur = int(durations[i, j].item())
                if dur > 0:
                    alignment[pos:pos+dur, j] = 1.0 / dur  # Normalize by duration
                    pos += dur
            
            alignments.append(alignment)
        
        # Stack across batch dimension
        return torch.stack(alignments, dim=0)
    
    def _expand_phonemes(self, phoneme_features, durations):
        """
        Expand phoneme features according to their durations
        
        Args:
            phoneme_features: Tensor of shape [batch_size, seq_len, dim]
            durations: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Expanded features of shape [batch_size, mel_len, dim]
        """
        batch_size, seq_len, dim = phoneme_features.shape
        
        # Convert durations to integer
        durations = durations.long()
        mel_len = torch.sum(durations, dim=1)
        max_mel_len = torch.max(mel_len).item()
        
        # Create expanded features
        expanded = torch.zeros(batch_size, max_mel_len, dim, device=phoneme_features.device)
        
        for i in range(batch_size):
            pos = 0
            for j in range(seq_len):
                dur = durations[i, j].item()
                if dur > 0:
                    expanded[i, pos:pos+dur] = phoneme_features[i, j].unsqueeze(0).expand(dur, -1)
                    pos += dur
        
        return expanded
    
    def forward(self, phoneme_emb, durations, f0, speaker_emb, language_emb):
        """
        Generate mel-spectrogram from linguistic and acoustic features
        
        Args:
            phoneme_emb: Phoneme embeddings of shape [batch_size, seq_len, emb_dim]
            durations: Phoneme durations of shape [batch_size, seq_len]
            f0: Fundamental frequency contours of shape [batch_size, mel_len, 1]
            speaker_emb: Speaker embeddings of shape [batch_size, emb_dim]
            language_emb: Language embeddings of shape [batch_size, emb_dim]
            
        Returns:
            Predicted mel-spectrogram of shape [batch_size, mel_len, n_mels]
        """
        batch_size, seq_len = phoneme_emb.shape[0], phoneme_emb.shape[1]
        
        # Embed phoneme features
        phoneme_features = self.input_embedding(phoneme_emb)  # [batch, seq_len, model_dim]
        
        # Add duration information to phoneme features
        duration_emb = self.duration_adapter(durations.unsqueeze(-1))  # [batch, seq_len, model_dim]
        phoneme_features = phoneme_features + duration_emb
        
        # Encode phoneme sequence with transformer
        encoded_phonemes = self.encoder(phoneme_features)  # [batch, seq_len, model_dim]
        
        # Expand phoneme features according to durations
        expanded_features = self._expand_phonemes(encoded_phonemes, durations)  # [batch, mel_len, model_dim]
        
        # Get mel sequence length
        mel_len = f0.shape[1]
        
        # Process speaker embedding
        speaker_cond = self.speaker_adapter(speaker_emb)  # [batch, 256]
        
        # Process language embedding
        language_cond = self.language_adapter(language_emb)  # [batch, 256]
        
        # Process F0
        # Ensure F0 has the correct format
        if f0.dim() == 2:
            f0 = f0.unsqueeze(-1)  # [batch, mel_len, 1]
        
        # For efficient adaptation, we first process a single F0 vector for the sequence
        mean_f0 = torch.mean(f0, dim=1)  # [batch, 1]
        
        # Create an F0 feature vector by concatenating mean, min, max, etc.
        f0_min, _ = torch.min(f0, dim=1)
        f0_max, _ = torch.max(f0, dim=1)
        f0_range = f0_max - f0_min
        f0_features = torch.cat([mean_f0, f0_min, f0_max, f0_range], dim=-1)  # [batch, 4]
        
        # Expand to expected input dimension
        f0_features_expanded = F.pad(f0_features, (0, 256 - f0_features.shape[-1]))
        
        # Apply F0 adaptation
        f0_cond = self.f0_adapter(f0_features_expanded)  # [batch, 256]
        
        # Combine all conditioning vectors
        combined_cond = speaker_cond + language_cond + f0_cond  # [batch, 256]
        
        # Expand to match sequence length
        cond_expanded = combined_cond.unsqueeze(1).expand(-1, mel_len, -1)  # [batch, mel_len, 256]
        
        # Add conditioning to expanded features
        conditioned_features = torch.cat([expanded_features, cond_expanded], dim=-1)
        
        # Project back to model dimension if needed
        if conditioned_features.shape[-1] != self.model_dim:
            conditioned_features = F.pad(
                conditioned_features, 
                (0, self.model_dim - conditioned_features.shape[-1])
            )
        
        # Apply cross-attention between expanded features and original phoneme features
        cross_attn_features = conditioned_features
        for cross_attn_layer in self.cross_attention_layers:
            cross_attn_features = cross_attn_layer(
                cross_attn_features,  # Query
                encoded_phonemes      # Key/Value
            )
        
        # Track formants for preservation
        formant_features = self.formant_tracker(cross_attn_features)
        
        # Apply formant preservation
        preserved_features = self.formant_preserver(cross_attn_features)
        
        # Combine formant features with main features
        final_features = cross_attn_features + preserved_features.repeat(1, 1, 3)[:, :, :cross_attn_features.shape[-1]]
        
        # Project to mel-spectrogram
        mel_pred = self.output_proj(final_features)
        
        return mel_pred