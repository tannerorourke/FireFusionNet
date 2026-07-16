from typing import Dict
import torch
import torch.nn as nn
from .modules import (
    SpatialEncoder, 
    WindowedSpatialAttention, ChannelMixingAttention, TemporalMixingAttention, 
    BiHeadDecoder
)


class FireFusionModel(nn.Module):
    """
    Given feature channels (C) and timesteps (T), compute the risk of wildfire ignition across a H x W grid.

    Steps:
        - Encode (downsample) Spatial patterns with basic ResNet MLP-style CNN encoder
        - Run self-attention over larger HxW windows than Encoder (these generalize well)
        - Run self-attention over channels (features)
        - Run self-attention over time
        - Decode (upsample) into a (B, 1, H, W) grid
    """
    def __init__(self, in_channels, mp: Dict):
        super().__init__()
        ws_params   =mp["win_spatial_mixing"]
        cm_params   =mp["channel_mixing"]
        tm_params   =mp["temporal_mixing"]
        
        embed_dim   =mp["embed_dim"]

        ws_heads    =ws_params['num_heads']
        ws_win_size =ws_params['window_size']
        ws_dropout  =ws_params['dropout'];

        cm_heads    =cm_params['num_heads']
        cm_d_model  =cm_params['d_model']
        cm_mlp_ratio=cm_params['mlp_ratio']
        cm_dropout  =cm_params['dropout']

        tm_heads    =tm_params['num_heads']
        tm_mlp_ratio=tm_params['mlp_ratio']
        tm_dropout  =tm_params['dropout']

        out_size    =mp['out_size'];
        n_causes    =mp['n_cause_classes']

        self.encoder = SpatialEncoder(in_channels, embed_dim)
        self.ws_attn = WindowedSpatialAttention(embed_dim, num_heads=ws_heads, window_size=ws_win_size, dropout=ws_dropout)
        self.cm_attn = ChannelMixingAttention(num_heads=cm_heads, num_channels=embed_dim, d_model=cm_d_model, mlp_ratio=cm_mlp_ratio, dropout=cm_dropout)
        self.tm_attn = TemporalMixingAttention(embed_dim, num_heads=tm_heads, mlp_ratio=tm_mlp_ratio, dropout=tm_dropout)
        self.decoder = BiHeadDecoder(embed_dim, out_size, n_cause_classes=n_causes)

    def forward(self, x: torch.Tensor):
        y = self.encoder(x)
        # print(f"[WFM] Encoding complete...")
        y = self.ws_attn(y)
        # print(f"[WFM] Windowed Spatial Attention complete...")
        y = self.cm_attn(y)
        # print(f"[WFM] Channel Mixing Attention complete...")
        y = self.tm_attn(y)
        # print(f"[WFM] Temporal Mixing Attention complete...")

        # Only decode the prediction from the last day
        y = y[:, -1]

        outputs = self.decoder(y)
        return outputs


