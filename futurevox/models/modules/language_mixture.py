import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageMixtureModule(nn.Module):
    """
    Language Mixture Module for FutureVox+
    
    This module handles:
    1. Cross-lingual phoneme mapping
    2. Accent continuum modeling
    3. Language-specific prosody and articulation
    4. Accent strength control
    
    Total parameters: ~2.1M
    """
    
    def __init__(self, embedding_dim, num_languages, num_phonemes, hidden_dims, accent_strength_dims):
        """
        Initialize language mixture module
        
        Args:
            embedding_dim: Dimension of language embeddings
            num_languages: Number of languages in the embedding table
            num_phonemes: Number of phonemes in the universal phoneme set
            hidden_dims: List of hidden dimensions for transformation networks
            accent_strength_dims: List of hidden dimensions for accent strength control
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_languages = num_languages
        self.num_phonemes = num_phonemes
        
        # Phoneme embedding table (~0.1M parameters)
        self.phoneme_embeddings = nn.Embedding(num_phonemes, embedding_dim)
        
        # Language embedding table (~0.05M parameters)
        self.language_embeddings = nn.Embedding(num_languages, embedding_dim)
        
        # Cross-lingual transformation (~0.4M parameters)
        self.xling_transform = self._build_mlp(
            embedding_dim * 2,  # Concatenated phoneme + language
            embedding_dim,
            hidden_dims
        )
        
        # Accent strength control (~0.1M parameters)
        self.accent_control = self._build_mlp(
            embedding_dim,
            accent_strength_dims[-1],
            accent_strength_dims[:-1]
        )
        
        # Language-specific prosody (~0.25M parameters)
        self.prosody_net = self._build_mlp(
            embedding_dim,
            embedding_dim,
            hidden_dims[:2]  # Use first two hidden dimensions
        )
        
        # Accent interpolation network (~0.4M parameters)
        self.accent_interp = self._build_mlp(
            embedding_dim * 2,  # Two language embeddings
            embedding_dim,
            hidden_dims
        )
        
        # Context-dependent adaptation (~0.4M parameters)
        self.context_adapt = self._build_mlp(
            embedding_dim * 2,  # Language + phonetic context
            embedding_dim,
            hidden_dims[:2]
        )
        
        # Language-specific articulation modeling (~0.4M parameters)
        self.articulation = self._build_mlp(
            embedding_dim * 2,  # Language + phoneme
            embedding_dim,
            hidden_dims
        )
        
        # Initialize weights
        self._init_weights()
    
    def _build_mlp(self, input_dim, output_dim, hidden_dims):
        """
        Build a multi-layer perceptron
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden dimensions
            
        Returns:
            Sequential MLP model
        """
        layers = []
        current_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.LeakyReLU(0.1))
            current_dim = dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """
        Initialize network weights for better training
        """
        # Initialize embedding tables with normal distribution
        nn.init.normal_(self.phoneme_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.language_embeddings.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, language_id_mixture, phonemes):
        """
        Process language ID mixture and phonemes
        
        Args:
            language_id_mixture: Tensor of shape [batch_size, num_languages, 2]
                where [:, :, 0] contains language IDs and [:, :, 1] contains mixture weights
            phonemes: Tensor of shape [batch_size, seq_len]
                
        Returns:
            Tuple of (language_embedding, phoneme_embedding)
                language_embedding: Tensor of shape [batch_size, embedding_dim]
                phoneme_embedding: Tensor of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size = language_id_mixture.shape[0]
        
        # Extract language IDs and weights
        language_ids = language_id_mixture[:, :, 0].long()  # [batch_size, num_languages]
        mixture_weights = language_id_mixture[:, :, 1]      # [batch_size, num_languages]
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights, dim=1)
        
        # Get embeddings for each language
        language_embs = self.language_embeddings(language_ids)  # [batch_size, num_languages, embedding_dim]
        
        # Apply mixture weights
        weighted_embs = language_embs * mixture_weights.unsqueeze(-1)
        mixed_language = torch.sum(weighted_embs, dim=1)  # [batch_size, embedding_dim]
        
        # Get phoneme embeddings
        phoneme_embs = self.phoneme_embeddings(phonemes)  # [batch_size, seq_len, embedding_dim]
        
        # Apply cross-lingual transformation to each phoneme
        # Expand language embedding to match phoneme sequence length
        lang_expanded = mixed_language.unsqueeze(1).expand(-1, phoneme_embs.size(1), -1)
        
        # Concatenate phoneme and language embeddings
        combined = torch.cat([phoneme_embs, lang_expanded], dim=-1)
        
        # Apply cross-lingual transformation
        transformed_phonemes = self.xling_transform(combined)
        
        # Apply context-dependent adaptation
        # Get average phoneme context
        context_vector = torch.mean(phoneme_embs, dim=1)  # [batch_size, embedding_dim]
        
        # Concatenate language and context
        context_combined = torch.cat([mixed_language, context_vector], dim=-1)
        
        # Apply context adaptation
        adapted_language = self.context_adapt(context_combined)
        
        # Apply language-specific prosody
        lang_prosody = self.prosody_net(adapted_language)
        
        # Apply language-specific articulation
        # Expand adapted language to match phoneme sequence length
        lang_expanded = lang_prosody.unsqueeze(1).expand(-1, phoneme_embs.size(1), -1)
        
        # Concatenate transformed phonemes and language
        articulation_input = torch.cat([transformed_phonemes, lang_expanded], dim=-1)
        
        # Apply articulation transformation
        final_phonemes = self.articulation(articulation_input)
        
        return adapted_language, final_phonemes
    
    def interpolate_languages(self, language_ids, interpolation_weights):
        """
        Interpolate between multiple languages to create accent continuum
        
        Args:
            language_ids: Tensor of shape [batch_size, num_languages]
            interpolation_weights: Tensor of shape [batch_size, num_languages]
                Weights should sum to 1
                
        Returns:
            Interpolated language embedding
        """
        # Get language embeddings
        language_embs = self.language_embeddings(language_ids)  # [batch_size, num_languages, embedding_dim]
        
        # Normalize weights
        weights = F.softmax(interpolation_weights, dim=1)
        
        # Apply weights
        weighted_embs = language_embs * weights.unsqueeze(-1)
        mixed_language = torch.sum(weighted_embs, dim=1)  # [batch_size, embedding_dim]
        
        return mixed_language
    
    def adjust_accent_strength(self, source_language_id, target_language_id, accent_strength):
        """
        Adjust accent strength between source and target languages
        
        Args:
            source_language_id: Tensor of shape [batch_size]
            target_language_id: Tensor of shape [batch_size]
            accent_strength: Float between 0 and 1
                0 = pure target language, 1 = pure source language
                
        Returns:
            Language embedding with controlled accent strength
        """
        # Get language embeddings
        source_emb = self.language_embeddings(source_language_id)
        target_emb = self.language_embeddings(target_language_id)
        
        # Interpolate between embeddings
        mixed_emb = accent_strength * source_emb + (1 - accent_strength) * target_emb
        
        # Apply accent control network
        accent_params = self.accent_control(mixed_emb)
        
        # Concatenate embeddings for interpolation
        concat_embs = torch.cat([source_emb, target_emb], dim=-1)
        
        # Apply accent interpolation with learned parameters
        interpolated = self.accent_interp(concat_embs)
        
        # Weight the interpolation by accent strength
        final_emb = accent_strength * interpolated + (1 - accent_strength) * target_emb
        
        return final_emb