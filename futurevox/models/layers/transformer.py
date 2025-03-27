import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert self.head_dim * num_heads == model_dim, "model_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        
        # Output projection
        self.output_proj = nn.Linear(model_dim, model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key=None, value=None, mask=None):
        """
        Apply multi-head attention
        
        Args:
            query: Query tensor of shape [batch_size, query_length, model_dim]
            key: Key tensor of shape [batch_size, key_length, model_dim] (or None to use query)
            value: Value tensor of shape [batch_size, key_length, model_dim] (or None to use key)
            mask: Mask tensor of shape [batch_size, query_length, key_length] or [batch_size, 1, key_length]
            
        Returns:
            Output tensor of shape [batch_size, query_length, model_dim]
        """
        batch_size = query.shape[0]
        
        # Default to self-attention if key/value not provided
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_k, D]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, L_q, L_k]
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for broadcasting
            if mask.dim() == 3:  # [B, L_q, L_k]
                mask = mask.unsqueeze(1)  # [B, 1, L_q, L_k]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # [B, H, L_q, D]
        
        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)  # [B, L_q, D_model]
        
        # Final projection
        output = self.output_proj(context)
        
        return output


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, model_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Apply feed-forward network
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, model_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, model_dim]
        """
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer
    """
    def __init__(self, model_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Process through transformer encoder layer
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, model_dim]
            mask: Attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, model_dim]
        """
        # Self-attention with residual connection and normalization
        attn_output = self.self_attn(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple layers
    """
    def __init__(self, model_dim, num_layers, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x, mask=None):
        """
        Process through transformer encoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, model_dim]
            mask: Attention mask
            
        Returns:
            Output tensor of shape [batch_size, seq_length, model_dim]
        """
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class TransformerCrossAttention(nn.Module):
    """
    Transformer cross-attention layer
    """
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Apply cross-attention
        
        Args:
            query: Query tensor of shape [batch_size, query_length, model_dim]
            key_value: Key/value tensor of shape [batch_size, kv_length, model_dim]
            mask: Attention mask
            
        Returns:
            Output tensor of shape [batch_size, query_length, model_dim]
        """
        # Cross-attention with residual connection and normalization
        q = self.norm1(query)
        k = self.norm2(key_value)
        v = k
        
        attn_output = self.cross_attn(q, k, v, mask)
        output = query + self.dropout(attn_output)
        
        return output