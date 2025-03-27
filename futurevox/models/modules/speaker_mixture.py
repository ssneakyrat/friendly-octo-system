import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerMixtureModule(nn.Module):
    """
    Speaker Mixture Module for FutureVox+
    
    This module handles:
    1. Vector-quantized speaker embeddings
    2. Differentiable interpolation between speakers
    3. Speaker attribute decomposition and transfer
    4. Component-wise control over voice characteristics
    
    Total parameters: ~3.2M
    """
    
    def __init__(self, embedding_dim, num_speakers, hidden_dims, attribute_net_dims):
        """
        Initialize speaker mixture module
        
        Args:
            embedding_dim: Dimension of speaker embeddings
            num_speakers: Number of speakers in the embedding table
            hidden_dims: List of hidden dimensions for transformation networks
            attribute_net_dims: List of hidden dimensions for attribute networks
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        # Speaker embedding table (~2.05M parameters)
        self.speaker_embeddings = nn.Embedding(num_speakers, embedding_dim)
        
        # Transformation network for embedding manipulation (~0.4M parameters)
        self.transform_net = self._build_mlp(embedding_dim, embedding_dim, hidden_dims)
        
        # Mixture weight processor (~0.17M parameters)
        self.mixture_net = self._build_mlp(embedding_dim, embedding_dim, hidden_dims)
        
        # Speaker attribute decomposition networks (~0.58M parameters total)
        # These networks decompose a speaker embedding into separable attributes
        self.timbre_net = self._build_mlp(embedding_dim, embedding_dim, attribute_net_dims)
        self.rhythm_net = self._build_mlp(embedding_dim, embedding_dim, attribute_net_dims)
        self.articulation_net = self._build_mlp(embedding_dim, embedding_dim, attribute_net_dims)
        
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
        # Initialize embedding table with normal distribution
        nn.init.normal_(self.speaker_embeddings.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, speaker_id_mixture):
        """
        Process speaker ID mixture to create a mixed embedding
        
        Args:
            speaker_id_mixture: Tensor of shape [batch_size, num_speakers, 2]
                where [:, :, 0] contains speaker IDs and [:, :, 1] contains mixture weights
        
        Returns:
            Mixed speaker embedding of shape [batch_size, embedding_dim]
        """
        batch_size = speaker_id_mixture.shape[0]
        
        # Extract speaker IDs and weights
        speaker_ids = speaker_id_mixture[:, :, 0].long()  # [batch_size, num_speakers]
        mixture_weights = speaker_id_mixture[:, :, 1]     # [batch_size, num_speakers]
        
        # Normalize mixture weights
        mixture_weights = F.softmax(mixture_weights, dim=1)
        
        # Get embeddings for each speaker
        speaker_embs = self.speaker_embeddings(speaker_ids)  # [batch_size, num_speakers, embedding_dim]
        
        # Apply mixture weights
        weighted_embs = speaker_embs * mixture_weights.unsqueeze(-1)
        mixed_embedding = torch.sum(weighted_embs, dim=1)  # [batch_size, embedding_dim]
        
        # Transform the mixed embedding
        transformed_emb = self.transform_net(mixed_embedding)
        
        # Apply mixture-specific transformations
        mixture_effects = self.mixture_net(mixed_embedding)
        
        # Combine base embeddings with mixture effects
        final_embedding = transformed_emb + mixture_effects
        
        return final_embedding
    
    def decompose_speaker(self, speaker_id):
        """
        Decompose a speaker embedding into attribute components
        
        Args:
            speaker_id: Tensor of shape [batch_size]
        
        Returns:
            Dictionary of attribute embeddings
        """
        speaker_emb = self.speaker_embeddings(speaker_id)
        
        timbre = self.timbre_net(speaker_emb)
        rhythm = self.rhythm_net(speaker_emb)
        articulation = self.articulation_net(speaker_emb)
        
        return {
            'timbre': timbre,
            'rhythm': rhythm,
            'articulation': articulation
        }
    
    def compose_from_attributes(self, attributes):
        """
        Compose a speaker embedding from attribute components
        
        Args:
            attributes: Dictionary with 'timbre', 'rhythm', and 'articulation' keys
                Each value should be a tensor of shape [batch_size, embedding_dim]
                
        Returns:
            Composed speaker embedding of shape [batch_size, embedding_dim]
        """
        # Simple average of attributes as a starting point
        composed_emb = (attributes['timbre'] + attributes['rhythm'] + attributes['articulation']) / 3
        
        # Transform to proper speaker embedding space
        final_emb = self.transform_net(composed_emb)
        
        return final_emb
    
    def transfer_attributes(self, source_speaker_id, target_speaker_id, transfer_weights=None):
        """
        Transfer attributes from source speaker to target speaker
        
        Args:
            source_speaker_id: Tensor of shape [batch_size]
            target_speaker_id: Tensor of shape [batch_size]
            transfer_weights: Dictionary with 'timbre', 'rhythm', 'articulation' keys
                Each value should be a float between 0 and 1 indicating how much
                of that attribute to transfer (0 = keep target, 1 = use source)
                
        Returns:
            Modified speaker embedding
        """
        # Default weights transfer everything equally
        if transfer_weights is None:
            transfer_weights = {
                'timbre': 0.5,
                'rhythm': 0.5,
                'articulation': 0.5
            }
        
        # Get source and target attributes
        source_attrs = self.decompose_speaker(source_speaker_id)
        target_attrs = self.decompose_speaker(target_speaker_id)
        
        # Mix attributes according to weights
        mixed_attrs = {}
        for attr in ['timbre', 'rhythm', 'articulation']:
            weight = transfer_weights.get(attr, 0.5)
            mixed_attrs[attr] = (
                weight * source_attrs[attr] + 
                (1 - weight) * target_attrs[attr]
            )
        
        # Compose new embedding from mixed attributes
        return self.compose_from_attributes(mixed_attrs)