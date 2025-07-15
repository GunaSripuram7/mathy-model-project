"""
Model architecture definition for mathematical art generation.
Implements a diffusion-based model that can generate images from mathematical formulas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math
from typing import Dict, Optional, Tuple, List

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like architectures"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FormulaEncoder(nn.Module):
    """Encodes mathematical formulas into latent representations"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, max_len: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Additional layers for mathematical understanding
        self.math_type_classifier = nn.Linear(d_model, 10)  # trigonometric, polynomial, etc.
        self.complexity_predictor = nn.Linear(d_model, 1)
        
    def forward(self, formula_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Token embedding and positional encoding
        x = self.embedding(formula_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Apply transformer
        if attention_mask is not None:
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)
        
        # Global representation (mean pooling)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1, keepdim=True).float()
            global_repr = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            global_repr = x.mean(dim=1)
        
        # Additional predictions for mathematical understanding
        math_type_logits = self.math_type_classifier(global_repr)
        complexity_score = self.complexity_predictor(global_repr)
        
        return {
            'sequence_output': x,
            'pooled_output': global_repr,
            'math_type_logits': math_type_logits,
            'complexity_score': complexity_score
        }

class ResidualBlock(nn.Module):
    """Residual block for U-Net architecture"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.block2(h)
        
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    """Self-attention block for spatial attention"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        
        h = self.group_norm(x)
        q = self.q(h).view(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W).permute(0, 2, 1)
        
        # Compute attention
        attention = torch.bmm(q, k) / math.sqrt(C)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attention, v).permute(0, 2, 1).view(B, C, H, W)
        out = self.proj_out(out)
        
        return x + out

class CrossAttentionBlock(nn.Module):
    """Cross-attention between image features and formula representations"""
    
    def __init__(self, img_channels: int, formula_dim: int, num_heads: int = 8):
        super().__init__()
        
        self.img_channels = img_channels
        self.formula_dim = formula_dim
        self.num_heads = num_heads
        
        self.group_norm = nn.GroupNorm(8, img_channels)
        
        # Project image features to query
        self.to_q = nn.Conv2d(img_channels, img_channels, 1)
        
        # Project formula features to key and value
        self.to_k = nn.Linear(formula_dim, img_channels)
        self.to_v = nn.Linear(formula_dim, img_channels)
        
        self.proj_out = nn.Conv2d(img_channels, img_channels, 1)
        
    def forward(self, img_features: torch.Tensor, formula_features: torch.Tensor):
        B, C, H, W = img_features.shape
        
        # Normalize image features
        h = self.group_norm(img_features)
        
        # Get queries from image features
        q = self.to_q(h).view(B, C, H * W).permute(0, 2, 1)
        
        # Get keys and values from formula features
        k = self.to_k(formula_features).permute(0, 2, 1)  # B, C, seq_len
        v = self.to_v(formula_features).permute(0, 2, 1)  # B, C, seq_len
        
        # Compute cross-attention
        attention = torch.bmm(q, k) / math.sqrt(C)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(attention, v.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, H, W)
        out = self.proj_out(out)
        
        return img_features + out

class UNetWithFormulaCondition(nn.Module):
    """U-Net architecture with formula conditioning for diffusion model"""
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 formula_dim: int = 512,
                 time_emb_dim: int = 256,
                 channels: List[int] = [64, 128, 256, 512],
                 attention_resolutions: List[int] = [32, 16, 8]):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.formula_dim = formula_dim
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        
        # Formula conditioning
        self.formula_proj = nn.Linear(formula_dim, time_emb_dim)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        # Encoder (downsampling)
        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.down_cross_attentions = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip([channels[0]] + channels[:-1], channels)):
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch, out_ch, time_emb_dim * 4),
                ResidualBlock(out_ch, out_ch, time_emb_dim * 4),
            ]))
            
            # Add attention at certain resolutions
            if 64 // (2**i) in attention_resolutions:
                self.down_attentions.append(AttentionBlock(out_ch))
                self.down_cross_attentions.append(CrossAttentionBlock(out_ch, formula_dim))
            else:
                self.down_attentions.append(nn.Identity())
                self.down_cross_attentions.append(nn.Identity())
        
        # Middle block
        self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_emb_dim * 4)
        self.mid_attention = AttentionBlock(channels[-1])
        self.mid_cross_attention = CrossAttentionBlock(channels[-1], formula_dim)
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_emb_dim * 4)
        
        # Decoder (upsampling)
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.up_cross_attentions = nn.ModuleList()
        
        reversed_channels = list(reversed(channels))
        for i, (in_ch, out_ch) in enumerate(zip(reversed_channels, reversed_channels[1:] + [channels[0]])):
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim * 4),  # +out_ch for skip connection
                ResidualBlock(out_ch, out_ch, time_emb_dim * 4),
            ]))
            
            # Add attention at certain resolutions
            resolution = 8 * (2**i)
            if resolution in attention_resolutions:
                self.up_attentions.append(AttentionBlock(out_ch))
                self.up_cross_attentions.append(CrossAttentionBlock(out_ch, formula_dim))
            else:
                self.up_attentions.append(nn.Identity())
                self.up_cross_attentions.append(nn.Identity())
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1)
        )
    
    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal time embeddings"""
        half_dim = self.time_emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, 
                formula_condition: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.get_time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        
        # Formula conditioning
        formula_emb = self.formula_proj(formula_condition.mean(dim=1))  # Global pool
        time_emb = time_emb + formula_emb
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Encoder
        skip_connections = [x]
        for i, (blocks, attention, cross_attention) in enumerate(
            zip(self.down_blocks, self.down_attentions, self.down_cross_attentions)
        ):
            for block in blocks:
                x = block(x, time_emb)
            
            x = attention(x)
            if hasattr(cross_attention, 'forward') and not isinstance(cross_attention, nn.Identity):
                x = cross_attention(x, formula_condition)
            
            skip_connections.append(x)
            
            # Downsample
            if i < len(self.down_blocks) - 1:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        # Middle
        x = self.mid_block1(x, time_emb)
        x = self.mid_attention(x)
        x = self.mid_cross_attention(x, formula_condition)
        x = self.mid_block2(x, time_emb)
        
        # Decoder
        for i, (blocks, attention, cross_attention) in enumerate(
            zip(self.up_blocks, self.up_attentions, self.up_cross_attentions)
        ):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Skip connection
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            
            for block in blocks:
                x = block(x, time_emb)
            
            x = attention(x)
            if hasattr(cross_attention, 'forward') and not isinstance(cross_attention, nn.Identity):
                x = cross_attention(x, formula_condition)
        
        # Output
        x = self.conv_out(x)
        
        return x

class MathematicalArtModel(nn.Module):
    """Complete model for generating mathematical art from formulas"""
    
    def __init__(self, 
                 vocab_size: int = 10000,
                 formula_dim: int = 512,
                 image_size: int = 512,
                 **kwargs):
        super().__init__()
        
        self.formula_encoder = FormulaEncoder(vocab_size, formula_dim)
        self.unet = UNetWithFormulaCondition(formula_dim=formula_dim, **kwargs)
        
        # Additional components for training
        self.formula_reconstruction_head = nn.Linear(formula_dim, vocab_size)
        
    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor, 
                formula_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        
        # Encode formula
        formula_encoding = self.formula_encoder(formula_tokens, attention_mask)
        
        # Predict noise
        predicted_noise = self.unet(
            noisy_images, 
            timesteps, 
            formula_encoding['sequence_output']
        )
        
        return {
            'predicted_noise': predicted_noise,
            'formula_encoding': formula_encoding,
            'formula_reconstruction': self.formula_reconstruction_head(formula_encoding['pooled_output'])
        }

def create_model(config: Dict) -> MathematicalArtModel:
    """Factory function to create model from config"""
    return MathematicalArtModel(
        vocab_size=config.get('vocab_size', 10000),
        formula_dim=config.get('formula_dim', 512),
        image_size=config.get('image_size', 512),
        time_emb_dim=config.get('time_emb_dim', 256),
        channels=config.get('channels', [64, 128, 256, 512]),
        attention_resolutions=config.get('attention_resolutions', [32, 16, 8])
    )
