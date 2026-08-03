import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """ 
    Normalizer for the encoder, sized to the largest group count that divides `channels`.

    GroupNorm normalizes each sample independently and keeps no running
    statistics. BatchNorm's running mean/var would instead be an average over
    whatever extents were sampled during training, so training on crops drawn
    from one part of the domain would bake that region's statistics into the
    normalizer and apply them to every cell at full-domain inference.
    """
    return nn.GroupNorm(math.gcd(channels, max_groups), channels)


class ConvResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size = 3, stride = 1, padding = 1, dropout = 0.0):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.norm1 = group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=1, bias=False)
        self.norm2 = group_norm(out_ch)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)

        # only projected when the residual and trunk shapes disagree; an
        # unconditional branch would carry parameters that never see a gradient
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                group_norm(out_ch),
            )
            if stride != 1 or (in_ch != out_ch)
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out

class SpatialEncoder(nn.Module):
    """
    CNN with Residual Blocks over (H x W), extracting spatial features
    per time step T (we call H' and W')

    Shape: (B, T, C, H, W) --> (B, T, embed_dim, H', W')

    `depth`: num of stride-2 stages, each doubling RF (in grid cells)
             A dataset at 1/2 ground resolution needs one more stage to reach 
             the same distance in kilometres. Holding that distance fixed is what 
             makes results comparable across resolutions.
    """
    def __init__(self, in_channels, embed_dim, depth: int = 1):
        super().__init__()

        self.base_ch            = 64
        self.depth              = depth
        self.down1_dropout      = 0.01
        self.down2_dropout      = 0.01

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.base_ch, kernel_size=3, padding=1, bias=False),
            group_norm(self.base_ch),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout),
            ConvResidualBlock(self.base_ch, self.base_ch, stride=1, dropout=self.down1_dropout)
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                ConvResidualBlock(self.base_ch if i == 0 else embed_dim, embed_dim,
                                  stride=2, dropout=self.down2_dropout),
                ConvResidualBlock(embed_dim, embed_dim, stride=1, dropout=self.down2_dropout)
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.shape

        # -- every day is encoded independently, so T rides along in the batch axis
        out = self.stem(x.reshape(B*T, C, H, W))
        out = self.down1(out)

        # -- the resolution entering each stage, kept for the decoder to fuse; only
        # -- the predicted day is retained, since that is all the decoder consumes
        skips = [out.view(B, T, *out.shape[1:])[:, -1]]
        for stage in self.stages[:-1]:
            out = stage(out)
            skips.append(out.view(B, T, *out.shape[1:])[:, -1])
        out = self.stages[-1](out)

        e_dim, Hp, Wp = out.shape[1], out.shape[2], out.shape[3]
        return out.reshape(B, T, e_dim, Hp, Wp), skips


class WindowedSpatialAttention(nn.Module):
    """
    Windowed spatial self-attention, as discussed in https://arxiv.org/html/2306.08191v2
    Mixes Spatial attributes (H' x W') at a larger resolution than H' and W'

    Shape:  (B, T, C, H', W') --> (B, T, C, H', W') (no change)
    
    Tokens within a window carry a learned position: attention is permutation
    equivariant, so without one the block cannot tell upslope from downslope
    within its window, only that both are present.

    The feature map is padded up to a whole number of windows and cropped after,
    so window_size is a free hyperparameter rather than a divisor the grid has to
    satisfy. 
    Padded tokens masked out.
    """
    def __init__(self, embed_dim, num_heads, window_size, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.pos_embed = nn.Parameter(torch.randn(window_size * window_size, embed_dim) * 0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.window_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, Hp, Wp = x.shape
        ws = self.window_size

        x = x.permute(0, 1, 3, 4, 2).contiguous().view(B*T, Hp, Wp, C)

        pad_h = (ws - Hp % ws) % ws
        pad_w = (ws - Wp % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hpad, Wpad = Hp + pad_h, Wp + pad_w
        nH, nW = Hpad // ws, Wpad // ws

        x_windows = x.view(B*T, nH, ws, nW, ws, C).permute(0, 1, 3, 2, 4, 5)
        x_windows = x_windows.reshape(B*T*nH*nW, ws*ws, C)

        pad_mask = None
        if pad_h or pad_w:
            valid = torch.zeros(Hpad, Wpad, dtype=torch.bool, device=x.device)
            valid[:Hp, :Wp] = True
            valid = valid.view(nH, ws, nW, ws).permute(0, 2, 1, 3).reshape(nH*nW, ws*ws)
            # -- a window entirely outside the grid would mask every key and yield NaN
            valid[~valid.any(dim=1)] = True
            pad_mask = (~valid).repeat(B*T, 1)

        x_w = x_windows
        x_norm = self.norm(x_windows) + self.pos_embed
        # need_weights=False keeps the attention matrix unmaterialized
        # ++ lets torch dispatch its fused kernels; the weights are discarded regardless
        out, _ = self.window_attn(
            x_norm, x_norm, x_norm, need_weights=False, key_padding_mask=pad_mask
        )
        out = self.proj(out) + x_w

        out = out.view(B*T, nH, nW, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B*T, Hpad, Wpad, C)[:, :Hp, :Wp]
        return out.view(B, T, Hp, Wp, C).permute(0, 1, 4, 2, 3).contiguous()


class ChannelMixingAttention(nn.Module):
    """
    Multi-Head Attention over CHANNELS, for fixed (B, T, H', W')

    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W') (no change)
    Steps:
        - Tokenizes each channel into a d_model vector (value scale + identity)
        - Applies MH self-attention over channels
        - Applies MLP
        - Projects back to a scalar per channel

    Each (b, t, h', w') location is an independent attention problem, so the
    N = B*T*H'*W' locations are processed in chunks: this block lifts every
    channel to a d_model vector and is therefore d_model times wider than the
    residual stream around it, which otherwise sets the memory ceiling for the
    whole network. Chunking bounds that peak without altering the result.
    """
    # A chunk enters attention as (chunk_size, num_heads, C, d_head); the fused
    # kernels map that leading chunk_size*num_heads onto a CUDA grid dimension and
    # the launch fails outright past its cap -- hours into a run rather than at
    # startup. Measured on sm_86: 196608 launches, 262144 raises "invalid
    # configuration argument". The check runs while the model is built.
    MAX_ATTN_BATCH = 196608

    def __init__(self, num_channels, d_model, num_heads, mlp_ratio, dropout, chunk_size=4096):
        super().__init__()
        if chunk_size * num_heads > self.MAX_ATTN_BATCH:
            raise ValueError(
                f"channel_mixing chunk_size {chunk_size} x num_heads {num_heads} = "
                f"{chunk_size * num_heads} exceeds the {self.MAX_ATTN_BATCH} attention "
                f"batch limit; cap chunk_size at {self.MAX_ATTN_BATCH // num_heads}"
            )
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.dropout_p = dropout

        # Per-channel tokenizer: h[n, c, :] = x[n, c] * value_scale[c] + channel_embed[c].
        # A shared Linear(1, d_model) would map every channel through one vector,
        # leaving tokens without channel identity -- attention is permutation
        # equivariant, so two channels holding the same value would be
        # indistinguishable and no feature-pair interaction could be learned.
        self.value_scale = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)
        self.channel_embed = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, 1)

        self.norm1 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def _tokenize(self, x_chunk: torch.Tensor) -> torch.Tensor:
        # (n, embed_dim) -> (n, embed_dim, d_model)
        return x_chunk.unsqueeze(-1) * self.value_scale + self.channel_embed

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        # (n, embed_dim, d_model) -> (n, embed_dim, d_model)
        n, C, D = h.shape
        h_norm = self.norm1(h)

        qkv = self.qkv(h_norm).view(n, C, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # scaled_dot_product_attention keeps the (C, C) attention matrix out of
        # memory; nn.MultiheadAttention materializes it whenever weights are
        # requested, which is the default even when they are discarded
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        h = h + self.attn_proj(attn.transpose(1, 2).reshape(n, C, D))
        return h + self.mlp(self.norm2(h))

    def _block(self, x_chunk: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self._mix(self._tokenize(x_chunk))).squeeze(-1)

    def forward(self, x: torch.Tensor):
        # x: (B, T, embed_dim, H', W')
        B, T, embed_dim, Hp, Wp = x.shape
        assert embed_dim == self.num_channels, "ChannelMixBlock: num_channels doesn't match incoming embed_dim"

        # Move to (B*T*H'*W', embed_dim) to operate per spatial-temporal location
        x_flat = x.permute(0, 1, 3, 4, 2).reshape(B*T*Hp*Wp, embed_dim)

        recompute = self.training and torch.is_grad_enabled()
        outs = []
        for i in range(0, x_flat.shape[0], self.chunk_size):
            x_chunk = x_flat[i:i + self.chunk_size]
            # recomputing each chunk in backward keeps the widened tokens from
            # being retained for every chunk at once
            outs.append(
                checkpoint(self._block, x_chunk, use_reentrant=False)
                if recompute else self._block(x_chunk)
            )
        out_flat = torch.cat(outs, dim=0)

        # Reshape back to (B, T, embed_dim, H', W')
        return out_flat.view(B, T, Hp, Wp, embed_dim).permute(0, 1, 4, 2, 3).contiguous()


class TemporalMixingAttention(nn.Module):
    """
    Multi-Head Attention over time T, for fixed dims B, embed_dim, H', and W'

    Shape: (B, T, embed_dim, H', W') --> (B, T, embed_dim, H', W')

    Attention is permutation equivariant, so without a positional term the block
    reads the look-back window as an unordered bag of days: the same ten days in
    any order produce an identical output. Fire weather is a sequence -- a drying
    trend, a wind buildup, rain three days ago versus rain yesterday -- and none
    of that survives an order-blind pooling. A learned per-day embedding added to
    the residual stream restores day identity.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio, dropout, max_window: int = 64):
        super().__init__()
        # indexed from the end of the window, so position 0 is always the day
        # being predicted from regardless of how long the window is
        self.time_embed = nn.Parameter(torch.randn(max_window, embed_dim) * 0.02)
        self.max_window = max_window

        self.norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
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
        if T > self.max_window:
            raise ValueError(f"window of {T} days exceeds max_window={self.max_window}")

        # collapse B/H'/W' -- each pixel for each channel across time
        f_permute = f.permute(0, 3, 4, 1, 2).contiguous()
        x = f_permute.view(B*Hp*Wp, T, C)

        # -- reversed so the predicted-from day holds index 0 for any window length
        x = x + self.time_embed[:T].flip(0)

        # pre-norm: each sub-block normalizes its own input and leaves the
        # residual stream itself untouched, with a LayerNorm per sub-block
        x_norm = self.norm(x)
        out_attn, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + out_attn

        out_ffn = self.mlp(self.norm2(x))
        x = x + out_ffn

        # --> back to (B, T, embed_dim, H', W')
        out = x.view(B, Hp, Wp, T, C).permute(0, 3, 4, 1, 2).contiguous() 
        
        
        return out
    

class BiHeadDecoder(nn.Module):
    """
    Convert spatiotemporal features into H x W risk map.
    Input:  (B, embed_dim, H', W') -- time dimension collapsed to last day,
            plus the encoder's per-resolution feature maps for that day
    Output: (B, 1, H, W) and (B, num_classes, H, W) per two heads

    One upsample step inverts each encoder stride-2 stage, fusing the encoder
    map at that resolution. Interpolating straight from the deepest map would
    cap the output at the stride the encoder ended on; the fused maps are what
    let a fine-resolution run express structure below that stride.

    Each step resizes to the shape of the map it fuses rather than by a fixed
    factor, so odd extents survive the round trip and one model can train on
    crops and predict over the full domain.
    """
    def __init__(self, embed_dim, n_cause_classes: int, depth: int = 1,
                 base_ch: int = 64, head_ch: int = 64):
        super().__init__()
        self.n_cause_classes = n_cause_classes
        self.depth = depth

        skip_ch = [base_ch] + [embed_dim] * (depth - 1)
        in_ch = embed_dim
        fuse = []
        # -- built deepest first, matching the order the skips are consumed
        for level in reversed(range(depth)):
            out_ch = head_ch if level == 0 else embed_dim
            fuse.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch[level], out_ch, 3, padding=1),
                nn.GELU(),
            ))
            in_ch = out_ch
        self.fuse = nn.ModuleList(fuse)

        self.ignition_head = nn.Conv2d(head_ch, 1, kernel_size=1)

        self.cause_head = nn.Conv2d(head_ch, self.n_cause_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, skips):
        f = x
        for block, skip in zip(self.fuse, reversed(skips)):
            f = F.interpolate(f, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            f = block(torch.cat([f, skip], dim=1))

        ignition_logits = self.ignition_head(f) # (B, 1, H, W)
        cause_logits = self.cause_head(f)  # (B, num_classes, H, W)

        return ignition_logits, cause_logits
