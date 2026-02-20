import torch
import torch.nn as nn


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
class ConvResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size = 3, stride = 1, padding = 1, dropout = 0.0):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

        self.is_downsample = stride != 1 or (in_ch != out_ch)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)

        if self.is_downsample:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out

class SpatialEncoder(nn.Module):
    """
    CNN with Residual Blocks over (H x W), extracting spatial features 
    per time step T (we call H' and W')

    Shape: (B, T, C, H, W) --> (B, T, embed_dim, H', W')
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()

        """ Model Params """
        self.base_ch            = 64
        self.head_hidden_dim    = 312
        self.down1_dropout      = 0.01
        self.down2_dropout      = 0.01

        # In stem: downsample with 7x7 kernel + max pool ->> (B, 64, 24, 32)
        # Large kernel early to capture broader patterns, max pool down to kernel_size=3
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_ch),
            nn.ReLU(inplace=True)
        )

        #- Residual stage 1 (x2 blocks @ 64) ->> keep at (B, 64, 24, 32) res
        # Keep in/out channels to refine features
        self.down1 = nn.Sequential(
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout),
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout)
        )

        # downsample by factor of 2  |  (B, 64, 24, 32) ->> (B, 128, 12, 16)
        # First block downsamples, 2nd block refines using stride 1
        self.down2 = nn.Sequential(
            ConvResidualBlock(self.base_ch, embed_dim, stride=2, dropout=self.down2_dropout),
            ConvResidualBlock(embed_dim,    embed_dim, stride=1, dropout=self.down2_dropout)
        )

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape
        
        x = x.reshape(B*T, C, H, W) # merge T into batch
        out = self.stem(x)
        out = self.down1(out)
        # out = self.stem(out)
        out = self.down2(out)
        
        e_dim, Hp, Wp = out.shape[1], out.shape[2], out.shape[3]
        out = out.reshape(B, T, e_dim, Hp, Wp)
        return out

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class WindowedSpatialAttention(nn.Module):
    """
    Windowed spatial self-attention, as discussed in https://arxiv.org/html/2306.08191v2
    Mixes Spatial attributes (H' x W') at a larger resolution than H' and W'

    Shape:  (B, T, C, H', W') --> (B, T, C, H', W') (no change)

    Steps:
        - For each (B, T):
            - partition (H', W') into non-overlapping windows
            - run MultiheadAttention on flattened window sequences
            - permute back
    """
    def __init__(self, embed_dim, num_heads, window_size, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.norm = nn.LayerNorm(embed_dim)
        self.window_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H', W')
        B, T, C, Hp, Wp = x.shape
        ws = self.window_size

        assert Hp % ws == 0 and Wp % ws == 0, "H' and W' must be divisible by window_size"

        # Reshape to windows: (B*T * nH * nW, ws*ws, C)
        x = x.permute(0, 1, 3, 4, 2).contiguous() # Move channels last for easier shaping: (B, T, H', W', C)
        nH = Hp // ws
        nW = Wp // ws
        x_windows = x.view(B*T, nH, ws, nW, ws, C)          # (B*T, nH, ws, nW, ws, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5)     # (B*T, nH, nW, ws, ws, C)
        x_windows = x_windows.reshape(B*T*nH*nW, ws*ws, C)  # (num_windows, tokens, C)

        # residual connection for better grad flow
        x_w = x_windows
        x_norm = self.norm(x_windows)
        out, _ = self.window_attn(x_norm, x_norm, x_norm)  # (num_windows, tokens, C) -- Self-attention within each window
        out = self.proj(out)
        out = out + x_w

        # Reshape back to (B, T, C, H', W')
        out = out.view(B*T, nH, nW, ws, ws, C)             # (B*T, nH, nW, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5)                # (B*T, nH, ws, nW, ws, C)
        out = out.reshape(B*T, Hp, Wp, C)                  # (B*T, H', W', C)
        out = out.view(B, T, Hp, Wp, C).permute(0, 1, 4, 2, 3).contiguous()
        return out

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class ChannelMixingAttention(nn.Module):
    """
    Multi-Head Attention over CHANNELS, for fixed (B, T, H', W')
    
    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W') (no change)
    Steps:
        - Projects channel dimes (embed_dim) -> d_model
        - Applies MH self-attention over channels
        - Applies MLP
        - Projects back to embed_dim
    """
    def __init__(self, num_channels, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model

        
        self.in_proj = nn.Linear(1, d_model) # embed_dim -> d_model
        self.out_proj = nn.Linear(d_model, 1)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        hidden_dim = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, embed_dim, H', W')
        B, T, embed_dim, Hp, Wp = x.shape
        assert embed_dim == self.num_channels, "ChannelMixBlock: num_channels doesn't match incoming embed_dim"

        # Move to (B*T*H'*W', embed_dim) to operate per spatial-temporal location
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()      # (B, T, H', W', embed_dim)
        x_flat = x_perm.view(B*T*Hp*Wp, embed_dim)          # (N, embed_dim)

        # Add a dummy feature dim for per-channel projection: (N, embed_dim, 1)
        x_feat = x_flat.unsqueeze(-1)                       # (N, embed_dim, 1)

        # Project to d_model
        h = self.in_proj(x_feat)                            # (N, embed_dim, d_model)

        # Attention over channels
        # # sequence len = embed_dim, embed_dim = d_model
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)     # (N, embed_dim, d_model)
        h = h + attn_out                                    # residual

        # MLP
        h_norm2 = self.norm2(h)
        h_ffn = self.mlp(h_norm2)                           # (N, embed_dim, d_model)
        h = h + h_ffn                                       # residual

        # Project back to scalar per channel
        out_feat = self.out_proj(h)                         # (N, embed_dim, 1)
        out_flat = out_feat.squeeze(-1)                     # (N, embed_dim)

        # Reshape back to (B, T, embed_dim, H', W')
        out = out_flat.view(B, T, Hp, Wp, embed_dim).permute(0, 1, 4, 2, 3).contiguous()
        return out

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class TemporalMixingAttention(nn.Module):
    """
    Multi-Head Attention over TIME T, for fixed dims B, embed_dim, H', and W'
    
    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W')
    Steps:
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(p=dropout/2)
        )

    def forward(self, f):
        B, T, C, Hp, Wp = f.shape

        # collapse B/H'/W' -- each pixel for each channel across time
        f_permute = f.permute(0, 3, 4, 1, 2).contiguous() 
        x = f_permute.view(B*Hp*Wp, T, C)

        x = self.norm(x)
        out_attn, _ = self.attn(x, x, x)
        x = x + out_attn

        x = self.norm(x)
        out_ffn = self.mlp(x)
        x = x + out_ffn

        # --> back to (B, T, embed_dim, H', W')
        out = x.view(B, Hp, Wp, T, C).permute(0, 3, 4, 1, 2).contiguous() 
        
        
        return out
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class BiHeadDecoder(nn.Module):
    """
    Convert spatiotemporal features into H x W risk map.
    Input:  (B, embed_dim, H', W') -- time dimension collapsed to last day
    Output: (B, 1, H, W) and (B, num_classes, H, W) per two heads
    """
    def __init__(self, embed_dim, out_size, n_cause_classes = 3):
        super().__init__()
        self.out_H, self.out_W = out_size
        self.n_cause_classes = n_cause_classes

        self.shared_head = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(size=out_size, mode='bilinear')
        )

        self.ignition_head = nn.Conv2d(64, 1, kernel_size=1)

        self.cause_head = nn.Conv2d(64, self.n_cause_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # f: (B, embed_dim, H’, W’)
        f = self.shared_head(x)

        ignition_logits = self.ignition_head(f) # (B, 1, H, W)
        cause_logits = self.cause_head(f)  # (B, num_classes, H, W)    

        return ignition_logits, cause_logits