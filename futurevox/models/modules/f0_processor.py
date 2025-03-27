import torch
import torch.nn as nn
import torch.nn.functional as F


class F0ProcessorModule(nn.Module):
    """
    F0 Processing Module for FutureVox+
    
    This module handles:
    1. Register-aware F0 transformation with vocal register boundary detection
    2. Linguistic-musical F0 decomposition for preserving language tonality
    3. Multi-register processing for chest, mixed, and head voice
    4. Pitch adaptation between language-specific ranges
    
    Total parameters: ~1.75M
    """
    
    def __init__(self, input_dim, hidden_dims, num_registers=3, decomp_dims=None):
        """
        Initialize F0 processor module
        
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden dimensions for transformation networks
            num_registers: Number of voice registers (typically 3: chest, mixed, head)
            decomp_dims: Dimensions for decomposition networks
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_registers = num_registers
        
        if decomp_dims is None:
            decomp_dims = [512, 256]
        
        # Register boundary detection network (~0.11M parameters)
        self.register_detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dims[2], num_registers),
            nn.Softmax(dim=-1)
        )
        
        # Piecewise transformation networks for each register (~0.07M parameters)
        self.register_transforms = nn.ModuleList([
            self._build_mlp(input_dim, 1, hidden_dims[:3][::-1])  # Reverse order for smaller networks
            for _ in range(num_registers)
        ])
        
        # F0 decomposition network (~0.26M parameters)
        self.decomposition_net = self._build_mlp(
            input_dim, 
            decomp_dims[-1] * 2,  # Output both linguistic and musical components
            decomp_dims[:-1]
        )
        
        # Linguistic F0 processor (~0.1M parameters)
        self.linguistic_processor = self._build_mlp(
            input_dim, 
            hidden_dims[-1],
            hidden_dims[:-1]
        )
        
        # Musical F0 processor (~0.1M parameters)
        self.musical_processor = self._build_mlp(
            input_dim, 
            hidden_dims[-1],
            hidden_dims[:-1]
        )
        
        # F0 recombination network (~0.06M parameters)
        self.recombination_net = self._build_mlp(
            hidden_dims[-1] * 2,  # Linguistic + musical
            input_dim,
            hidden_dims[-2:-1]  # Use second-to-last hidden dim
        )
        
        # Register-specific transformations (~0.3M parameters)
        self.register_specific_nets = nn.ModuleList([
            self._build_mlp(input_dim, hidden_dims[1], hidden_dims[:1])
            for _ in range(num_registers)
        ])
        
        # Common processing network (~0.39M parameters)
        self.common_processor = self._build_mlp(
            input_dim * 2,  # F0 + speaker embedding
            input_dim,
            hidden_dims[:1]
        )
        
        # Register blending network (~0.1M parameters)
        self.register_blender = self._build_mlp(
            input_dim, 
            hidden_dims[1],
            hidden_dims[:1]
        )
        
        # Language-specific adaptation (~0.04M parameters)
        self.language_adapter = self._build_mlp(
            input_dim,
            hidden_dims[2],
            hidden_dims[1:2]
        )
        
        # Speaker-specific adaptation (~0.04M parameters)
        self.speaker_adapter = self._build_mlp(
            input_dim,
            hidden_dims[2],
            hidden_dims[1:2]
        )
        
        # Pitch baseline modeling (~0.04M parameters)
        self.pitch_baseline = self._build_mlp(
            input_dim * 2,  # Language + speaker
            hidden_dims[2],
            hidden_dims[1:2]
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
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def detect_registers(self, f0):
        """
        Detect voice register boundaries from F0 contour
        
        Args:
            f0: Tensor of shape [batch_size, time_steps, 1]
            
        Returns:
            Register probabilities of shape [batch_size, time_steps, num_registers]
        """
        # Create additional F0 features
        f0_log = torch.log(torch.clamp(f0, min=1e-5))
        
        # Concatenate with original F0
        f0_features = torch.cat([f0, f0_log], dim=-1)
        
        # Apply register detector across time
        batch_size, time_steps = f0.shape[0], f0.shape[1]
        f0_flat = f0_features.reshape(-1, f0_features.shape[-1])
        
        # Pad to input_dim if necessary
        if f0_flat.shape[-1] < self.input_dim:
            padding = torch.zeros(f0_flat.shape[0], self.input_dim - f0_flat.shape[-1], 
                                  device=f0.device)
            f0_flat = torch.cat([f0_flat, padding], dim=-1)
        else:
            f0_flat = f0_flat[:, :self.input_dim]
        
        # Apply register detector
        register_probs = self.register_detector(f0_flat)
        
        # Reshape back to batch and time dimensions
        register_probs = register_probs.reshape(batch_size, time_steps, self.num_registers)
        
        return register_probs
    
    def decompose_f0(self, f0, language_emb):
        """
        Decompose F0 into linguistic and musical components
        
        Args:
            f0: Tensor of shape [batch_size, time_steps, 1]
            language_emb: Tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Tuple of (linguistic_f0, musical_f0)
        """
        batch_size, time_steps = f0.shape[0], f0.shape[1]
        
        # Create F0 features
        f0_log = torch.log(torch.clamp(f0, min=1e-5))
        
        # Concatenate with original F0
        f0_features = torch.cat([f0, f0_log], dim=-1)
        
        # Pad to input_dim if necessary
        if f0_features.shape[-1] < self.input_dim:
            padding = torch.zeros(batch_size, time_steps, 
                                 self.input_dim - f0_features.shape[-1], 
                                 device=f0.device)
            f0_features = torch.cat([f0_features, padding], dim=-1)
        else:
            f0_features = f0_features[:, :, :self.input_dim]
        
        # Expand language embedding across time
        lang_expanded = language_emb.unsqueeze(1).expand(-1, time_steps, -1)
        
        # Combine F0 and language information
        combined = f0_features + lang_expanded[:, :, :f0_features.shape[-1]]
        
        # Decompose into linguistic and musical components
        decomp_flat = combined.reshape(-1, combined.shape[-1])
        decomposed = self.decomposition_net(decomp_flat)
        decomposed = decomposed.reshape(batch_size, time_steps, -1)
        
        # Split the output into linguistic and musical components
        split_point = decomposed.shape[-1] // 2
        linguistic_f0 = decomposed[:, :, :split_point]
        musical_f0 = decomposed[:, :, split_point:]
        
        return linguistic_f0, musical_f0
    
    def process_by_register(self, f0, register_probs):
        """
        Process F0 using register-specific transformations
        
        Args:
            f0: Tensor of shape [batch_size, time_steps, 1]
            register_probs: Tensor of shape [batch_size, time_steps, num_registers]
            
        Returns:
            Processed F0 of shape [batch_size, time_steps, 1]
        """
        batch_size, time_steps = f0.shape[0], f0.shape[1]
        
        # Create F0 features
        f0_log = torch.log(torch.clamp(f0, min=1e-5))
        
        # Concatenate with original F0
        f0_features = torch.cat([f0, f0_log], dim=-1)
        
        # Pad to input_dim if necessary
        if f0_features.shape[-1] < self.input_dim:
            padding = torch.zeros(batch_size, time_steps, 
                                 self.input_dim - f0_features.shape[-1], 
                                 device=f0.device)
            f0_features = torch.cat([f0_features, padding], dim=-1)
        else:
            f0_features = f0_features[:, :, :self.input_dim]
        
        # Flatten for processing
        f0_flat = f0_features.reshape(-1, f0_features.shape[-1])
        
        # Process through each register-specific network
        register_outputs = []
        for i in range(self.num_registers):
            reg_output = self.register_specific_nets[i](f0_flat)
            register_outputs.append(reg_output)
        
        # Stack outputs
        stacked_outputs = torch.stack(register_outputs, dim=1)  # [batch*time, num_registers, hidden_dim]
        
        # Reshape register probabilities to match
        register_probs_flat = register_probs.reshape(-1, self.num_registers).unsqueeze(-1)
        
        # Apply register probabilities
        weighted_outputs = stacked_outputs * register_probs_flat
        
        # Sum over registers
        merged = torch.sum(weighted_outputs, dim=1)  # [batch*time, hidden_dim]
        
        # Reshape back to batch and time dimensions
        processed = merged.reshape(batch_size, time_steps, -1)
        
        # Final blending
        f0_out = self.register_blender(processed)
        
        return f0_out
    
    def forward(self, f0, speaker_emb, language_emb):
        """
        Process F0 with respect to speaker and language characteristics
        
        Args:
            f0: Tensor of shape [batch_size, time_steps, 1]
            speaker_emb: Tensor of shape [batch_size, embedding_dim]
            language_emb: Tensor of shape [batch_size, embedding_dim]
            
        Returns:
            Processed F0 of shape [batch_size, time_steps, 1]
        """
        batch_size, time_steps = f0.shape[0], f0.shape[1]
        
        # Detect voice registers
        register_probs = self.detect_registers(f0)
        
        # Decompose F0 into linguistic and musical components
        linguistic_f0, musical_f0 = self.decompose_f0(f0, language_emb)
        
        # Process linguistic and musical components separately
        processed_linguistic = self.linguistic_processor(linguistic_f0.reshape(-1, linguistic_f0.shape[-1]))
        processed_musical = self.musical_processor(musical_f0.reshape(-1, musical_f0.shape[-1]))
        
        # Reshape back
        processed_linguistic = processed_linguistic.reshape(batch_size, time_steps, -1)
        processed_musical = processed_musical.reshape(batch_size, time_steps, -1)
        
        # Recombine components
        combined_features = torch.cat([processed_linguistic, processed_musical], dim=-1)
        recombined_f0 = self.recombination_net(combined_features.reshape(-1, combined_features.shape[-1]))
        recombined_f0 = recombined_f0.reshape(batch_size, time_steps, -1)
        
        # Apply register-specific processing
        register_processed = self.process_by_register(f0, register_probs)
        
        # Apply language and speaker adaptation
        lang_adapter_out = self.language_adapter(language_emb)
        speaker_adapter_out = self.speaker_adapter(speaker_emb)
        
        # Expand adapters to match time dimension
        lang_adapter_expanded = lang_adapter_out.unsqueeze(1).expand(-1, time_steps, -1)
        speaker_adapter_expanded = speaker_adapter_out.unsqueeze(1).expand(-1, time_steps, -1)
        
        # Combine register-processed F0 with language and speaker adaptations
        adapted_f0 = register_processed + lang_adapter_expanded + speaker_adapter_expanded
        
        # Generate pitch baseline from language and speaker
        baseline_input = torch.cat([language_emb, speaker_emb], dim=-1)
        pitch_baseline_out = self.pitch_baseline(baseline_input)
        pitch_baseline_expanded = pitch_baseline_out.unsqueeze(1).expand(-1, time_steps, -1)
        
        # Expand speaker embedding for common processing
        speaker_expanded = speaker_emb.unsqueeze(1).expand(-1, time_steps, -1)
        
        # Combine F0 with speaker information for common processing
        common_input = torch.cat([adapted_f0, speaker_expanded[:, :, :adapted_f0.shape[-1]]], dim=-1)
        
        # Apply common processing
        final_f0 = self.common_processor(common_input)
        
        # Project to single dimension if needed
        if final_f0.shape[-1] > 1:
            final_f0 = final_f0[:, :, 0:1]
        
        # Ensure positive F0 values
        final_f0 = F.softplus(final_f0)
        
        return final_f0