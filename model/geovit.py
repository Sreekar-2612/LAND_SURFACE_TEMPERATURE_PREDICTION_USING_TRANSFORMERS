"""
GeoViT: Geospatial Vision Transformer for pixel-wise LST regression.

Architecture (as specified in the paper):

Input: pixel feature vector x ∈ R^C  (C = 13 channels)

1. Linear Projection:  z₀ = Wₑ·x + bₑ         → R^dim (dim=128)
2. Positional Encoding: z₁ = z₀ + p             (p is learnable)
3. Transformer Encoder × depth (depth=4):
     a. LayerNorm
     b. Multi-Head Self-Attention (heads=4)
        Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
     c. Residual connection
     d. LayerNorm
     e. FFN: Linear(dim→mlp_dim) → GELU → Linear(mlp_dim→dim)
     f. Residual connection
4. LayerNorm (final)
5. Linear head: R^dim → R^1  (single LST value)

Note on "patch-based" processing:
  The paper mentions patch_size=16 in the context of spatial patch extraction
  from rasters. In pixel-wise mode (which is how training is described —
  each pixel becomes a sample), the "patch" is the pixel's feature vector.
  The patch_size=16 refers to the spatial context window used to extract
  neighbourhood statistics. In this implementation we support both modes:
  - pixel_mode: each pixel's 13-channel vector is one sample (used for training)
  - patch_mode: 16×16 spatial patches are processed together (used for inference)
  
  For simplicity and faithfulness to the paper's training description,
  we train in pixel_mode and infer in pixel_mode (process all pixels of a
  raster stack as a flat pixel matrix).
"""

import torch
import torch.nn as nn
import math
from config import MODEL_CONFIG


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention as in standard Transformer.
    Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
    """
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads   = heads
        self.dk      = dim // heads        # dimension per head
        self.scale   = math.sqrt(self.dk)  # √dₖ scaling factor
        
        self.to_qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: (B, dim) for pixel-wise; or (B, N, dim) for sequence.
        In pixel-mode x is (B, dim) — treat each pixel as a single-token sequence.
        """
        # Handle pixel-wise (B, dim) by adding sequence dim
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, dim)
            squeeze = True
        
        B, N, D = x.shape
        
        # Project to Q, K, V
        qkv = self.to_qkv(x)                          # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dk)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, heads, N, dk)
        Q, K, V = qkv.unbind(0)                        # each: (B, heads, N, dk)
        
        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) / self.scale  # (B, heads, N, N)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted values
        out = attn @ V                                  # (B, heads, N, dk)
        out = out.permute(0, 2, 1, 3).reshape(B, N, D) # (B, N, dim)
        out = self.out_proj(out)
        
        if squeeze:
            out = out.squeeze(1)  # (B, dim)
        
        # Return attention weights for XAI (last head averaged)
        attn_weights = attn.mean(dim=1)  # (B, N, N)
        return out, attn_weights


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block:
    Pre-LayerNorm architecture (norm before attention and FFN).
    """
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1   = nn.LayerNorm(dim)
        self.attn    = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2   = nn.LayerNorm(dim)
        self.ffn     = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),                  # GELU activation (standard ViT)
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # Sub-layer 1: Self-attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = residual + attn_out
        
        # Sub-layer 2: FFN with residual
        residual = x
        x = residual + self.ffn(self.norm2(x))
        
        return x, attn_weights


class GeoViT(nn.Module):
    """
    Geospatial Vision Transformer for pixel-wise LST regression.
    
    Processes each pixel's 13-channel feature vector independently,
    using transformer self-attention to learn nonlinear feature interactions.
    
    Architecture matches Table 4 hyperparameters from the paper exactly.
    """
    def __init__(self, config=None):
        super().__init__()
        cfg = config or MODEL_CONFIG
        
        self.input_channels = cfg["input_channels"]   # 13
        self.dim            = cfg["dim"]              # 128
        self.depth          = cfg["depth"]            # 4
        self.heads          = cfg["heads"]            # 4
        self.mlp_dim        = cfg["mlp_dim"]          # 256
        self.dropout_rate   = cfg["dropout"]          # 0.0
        
        # 1. Linear Projection: input → embedding
        # z₀ = Wₑ · x + bₑ
        self.input_projection = nn.Linear(self.input_channels, self.dim)
        
        # 2. Positional Encoding (learnable, single pixel token)
        # p ∈ R^dim, added to the embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.dim) * 0.02)
        
        # 3. Transformer Encoder Blocks × depth
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(self.dim, self.heads, self.mlp_dim, self.dropout_rate)
            for _ in range(self.depth)
        ])
        
        # 4. Final Layer Norm
        # z_norm = LayerNorm(z_L)
        self.final_norm = nn.LayerNorm(self.dim)
        
        # 5. Regression Head: dim → 1
        # ŷ = Wₒᵀ · z_norm + bₒ
        self.regression_head = nn.Linear(self.dim, cfg["output_dim"])
        
        # Store attention weights for XAI
        self.attention_weights = []
        
        # Weight initialisation
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x:                (B, 13) normalised feature vectors
            return_attention: if True, also return attention weights for XAI
        
        Returns:
            ŷ: (B, 1) predicted normalised LST
        """
        # Step 1: Linear Projection → z₀
        z = self.input_projection(x)      # (B, dim)
        
        # Step 2: Add positional embedding → z₁
        z = z + self.pos_embedding         # (B, dim), broadcast over batch
        
        # Step 3: Transformer Encoder Blocks
        self.attention_weights = []
        for block in self.encoder_blocks:
            z, attn_w = block(z)
            self.attention_weights.append(attn_w)
        
        # Step 4: Final LayerNorm → z_norm
        z = self.final_norm(z)             # (B, dim)
        
        # Step 5: Regression Head → ŷ
        out = self.regression_head(z)      # (B, 1)
        
        if return_attention:
            return out, self.attention_weights
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
