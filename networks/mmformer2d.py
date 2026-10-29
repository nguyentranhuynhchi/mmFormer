"""mmFormer 2D Architecture for Multimodal Cardiac Pathology Segmentation (MyoPS-380).
Adapted from YaoZhang93/mmFormer (MICCAI 2022) to 2D multi-slice formulation.
Modalities: 3 (bSSFP, LGE, T2w)
Spatial Resolution: 128 x 128 (Downsampled to 8x8 tokens, 64 tokens per modality)
Classes: 4 Canonical (0: Background, 1: Normal Myocardium, 2: Edema, 3: Scar)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 3
patch_size = 8  # 128 / (2**4) = 8 -> 8x8 = 64 spatial tokens per modality


def normalization2d(planes: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm2d(planes)
    elif norm == "in":
        return nn.InstanceNorm2d(planes, affine=True)
    elif norm == "gn":
        return nn.GroupNorm(4, planes)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")


class general_conv2d_prenorm(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k_size, stride=stride,
            padding=padding, padding_mode=pad_type, bias=True
        )
        self.norm = normalization2d(in_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class general_conv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pad_type: str = "reflect",
        norm: str = "in",
        act_type: str = "lrelu",
        relufactor: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k_size, stride=stride,
            padding=padding, padding_mode=pad_type, bias=True
        )
        self.norm = normalization2d(out_ch, norm=norm)
        if act_type == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif act_type == "lrelu":
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class fusion_prenorm2d(nn.Module):
    def __init__(self, in_channel: int, num_modals: int = 3):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            general_conv2d_prenorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv2d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv2d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_layer(x)


class Encoder2D(nn.Module):
    """Hybrid CNN Intra-modal Encoder."""

    def __init__(self):
        super().__init__()
        self.e1_c1 = nn.Conv2d(1, basic_dims, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.e1_c2 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type="reflect")
        self.e1_c3 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type="reflect")

        self.e2_c1 = general_conv2d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type="reflect")
        self.e2_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type="reflect")
        self.e2_c3 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type="reflect")

        self.e3_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type="reflect")
        self.e3_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type="reflect")
        self.e3_c3 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type="reflect")

        self.e4_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type="reflect")
        self.e4_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type="reflect")
        self.e4_c3 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type="reflect")

        self.e5_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type="reflect")
        self.e5_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type="reflect")
        self.e5_c3 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type="reflect")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder_sep2D(nn.Module):
    """Separate auxiliary decoder for each individual modality."""

    def __init__(self, num_cls: int = 4):
        super().__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.d4_c1 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="reflect")

        self.d3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.d3_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.d2_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.d1_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv2d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_layer = nn.Conv2d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5) -> torch.Tensor:
        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        return self.softmax(logits)


class Decoder_fuse2D(nn.Module):
    """Fused progressive decoder with deep supervision heads."""

    def __init__(self, num_cls: int = 4, num_modals: int = 3):
        super().__init__()
        self.d4_c1 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type="reflect")
        self.d4_out = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type="reflect")

        self.d3_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type="reflect")
        self.d3_out = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type="reflect")

        self.d2_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type="reflect")
        self.d2_out = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type="reflect")

        self.d1_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type="reflect")
        self.d1_out = general_conv2d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type="reflect")

        self.seg_d4 = nn.Conv2d(basic_dims * 16, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv2d(basic_dims * 8, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv2d(basic_dims * 4, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(basic_dims * 2, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv2d(basic_dims, num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)

        self.RFM5 = fusion_prenorm2d(in_channel=basic_dims * 16, num_modals=num_modals)
        self.RFM4 = fusion_prenorm2d(in_channel=basic_dims * 8, num_modals=num_modals)
        self.RFM3 = fusion_prenorm2d(in_channel=basic_dims * 4, num_modals=num_modals)
        self.RFM2 = fusion_prenorm2d(in_channel=basic_dims * 2, num_modals=num_modals)
        self.RFM1 = fusion_prenorm2d(in_channel=basic_dims * 1, num_modals=num_modals)

    def forward(self, x1, x2, x3, x4, x5) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, qkv_bias: bool = False, qk_scale=None, dropout_rate: float = 0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim: int, dropout_rate: float, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, depth: int, heads: int, mlp_dim: int, dropout_rate: float = 0.1, n_levels: int = 1):
        super().__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList([
            Residual(PreNormDrop(embedding_dim, dropout_rate, SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate)))
            for _ in range(self.depth)
        ])
        self.cross_ffn_list = nn.ModuleList([
            Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate)))
            for _ in range(self.depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal2D(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, K, C, H, W), mask: (B, K)
        B, K, C, H, W = x.size()
        mask_expanded = mask.view(B, K, 1, 1, 1).float()
        y = x * mask_expanded
        return y.view(B, -1, H, W)


class mmFormer2D(nn.Module):
    """Full mmFormer model for 2D 3-modality cardiac pathology segmentation."""

    def __init__(self, num_cls: int = 4, num_modals: int = 3):
        super().__init__()
        self.num_cls = num_cls
        self.num_modals = num_modals

        # 3 Independent Intra-modal Encoders: bSSFP, LGE, T2w
        self.bssfp_encoder = Encoder2D()
        self.lge_encoder = Encoder2D()
        self.t2w_encoder = Encoder2D()

        # IntraFormer convolutions & positional tokens
        self.bssfp_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1)
        self.lge_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1)
        self.t2w_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1)

        self.bssfp_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1)
        self.lge_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1)
        self.t2w_decode_conv = nn.Conv2d(transformer_basic_dims, basic_dims * 16, kernel_size=1)

        num_tokens = patch_size * patch_size  # 8x8 = 64
        self.bssfp_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))
        self.lge_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))
        self.t2w_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))

        self.bssfp_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.lge_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2w_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)

        # InterFormer (Cross-modal transformer fusing 3 modalities: 3 x 64 = 192 tokens)
        self.multimodal_transformer = Transformer(
            embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals
        )
        self.multimodal_decode_conv = nn.Conv2d(
            transformer_basic_dims * num_modals, basic_dims * 16 * num_modals, kernel_size=1
        )

        self.masker = MaskModal2D()
        self.decoder_fuse = Decoder_fuse2D(num_cls=num_cls, num_modals=num_modals)
        self.decoder_sep = Decoder_sep2D(num_cls=num_cls)

        self.is_training = False

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, 3, H, W)
        B, _, H, W = x.size()
        if mask is None:
            mask = torch.ones((B, self.num_modals), dtype=torch.bool, device=x.device)

        # Feature extraction from 3 independent encoders
        bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5 = self.bssfp_encoder(x[:, 0:1, :, :])
        lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.lge_encoder(x[:, 1:2, :, :])
        t2w_x1, t2w_x2, t2w_x3, t2w_x4, t2w_x5 = self.t2w_encoder(x[:, 2:3, :, :])

        # IntraFormer processing
        bssfp_tok = self.bssfp_encode_conv(bssfp_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)
        lge_tok = self.lge_encode_conv(lge_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)
        t2w_tok = self.t2w_encode_conv(t2w_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)

        bssfp_intra_tok = self.bssfp_transformer(bssfp_tok, self.bssfp_pos)
        lge_intra_tok = self.lge_transformer(lge_tok, self.lge_pos)
        t2w_intra_tok = self.t2w_transformer(t2w_tok, self.t2w_pos)

        bssfp_intra_x5 = bssfp_intra_tok.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()
        lge_intra_x5 = lge_intra_tok.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()
        t2w_intra_x5 = t2w_intra_tok.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()

        if self.is_training:
            bssfp_pred = self.decoder_sep(bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5)
            lge_pred = self.decoder_sep(lge_x1, lge_x2, lge_x3, lge_x4, lge_x5)
            t2w_pred = self.decoder_sep(t2w_x1, t2w_x2, t2w_x3, t2w_x4, t2w_x5)

        # Modality masking
        x1 = self.masker(torch.stack((bssfp_x1, lge_x1, t2w_x1), dim=1), mask)
        x2 = self.masker(torch.stack((bssfp_x2, lge_x2, t2w_x2), dim=1), mask)
        x3 = self.masker(torch.stack((bssfp_x3, lge_x3, t2w_x3), dim=1), mask)
        x4 = self.masker(torch.stack((bssfp_x4, lge_x4, t2w_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((bssfp_intra_x5, lge_intra_x5, t2w_intra_x5), dim=1), mask)

        # InterFormer processing
        b_x5, l_x5, t_x5 = torch.chunk(x5_intra, self.num_modals, dim=1)
        multimodal_token_x5 = torch.cat((
            b_x5.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
            l_x5.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
            t_x5.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
        ), dim=1)
        multimodal_pos = torch.cat((self.bssfp_pos, self.lge_pos, self.t2w_pos), dim=1)
        multimodal_inter_tok = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)

        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_tok.view(B, patch_size, patch_size, transformer_basic_dims * self.num_modals).permute(0, 3, 1, 2).contiguous()
        )

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, multimodal_inter_x5)

        if self.is_training:
            return fuse_pred, (bssfp_pred, lge_pred, t2w_pred), preds
        return fuse_pred
