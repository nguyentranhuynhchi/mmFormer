import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import general_conv2d_prenorm, fusion_prenorm
except ImportError:
    from layers import general_conv2d_prenorm, fusion_prenorm

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 3
patch_size = 8


class Encoder(nn.Module):
    def __init__(self, basic_dims=8):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv2d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv2d_prenorm(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv2d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')
        self.e2_c3 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='reflect')

        self.e3_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')
        self.e3_c3 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='reflect')

        self.e4_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')
        self.e4_c3 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='reflect')

        self.e5_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')
        self.e5_c3 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='reflect')

    def forward(self, x):
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


class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4, basic_dims=8):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d4_c1 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d3_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d2_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.d1_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv2d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
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
        pred = self.softmax(logits)

        return pred


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4, num_modals=3, basic_dims=8):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_c2 = general_conv2d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='reflect')
        self.d4_out = general_conv2d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='reflect')

        self.d3_c1 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_c2 = general_conv2d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='reflect')
        self.d3_out = general_conv2d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_c2 = general_conv2d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='reflect')
        self.d2_out = general_conv2d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv2d_prenorm(basic_dims * 2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv2d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv2d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv2d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv2d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv2d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv2d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_cls=num_modals)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_cls=num_modals)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_cls=num_modals)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_cls=num_modals)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_cls=num_modals)

    def forward(self, x1, x2, x3, x4, x5):
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
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
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

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = []
        self.cross_ffn_list = []
        self.depth = depth
        for _ in range(self.depth):
            self.cross_attention_list.append(
                Residual(
                    PreNormDrop(
                        embedding_dim,
                        dropout_rate,
                        SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                    )
                )
            )
            self.cross_ffn_list.append(
                Residual(
                    PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate))
                )
            )

        self.cross_attention_list = nn.ModuleList(self.cross_attention_list)
        self.cross_ffn_list = nn.ModuleList(self.cross_ffn_list)

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask=None):
        # x: (B, K, C, H, W)
        B, K, C, H, W = x.size()
        if mask is not None:
            m = mask.view(B, K, 1, 1, 1)
            y = torch.zeros_like(x)
            y = torch.where(m, x, y)
            return y.view(B, -1, H, W)
        return x.view(B, -1, H, W)


class Model(nn.Module):
    def __init__(self, num_cls=4, num_modals=3):
        super(Model, self).__init__()
        self.num_modals = num_modals
        self.num_cls = num_cls

        # 3 Hybrid CNN Encoders for bSSFP (0), LGE (1), T2w (2)
        self.bssfp_encoder = Encoder(basic_dims=basic_dims)
        self.lge_encoder = Encoder(basic_dims=basic_dims)
        self.t2w_encoder = Encoder(basic_dims=basic_dims)

        ########### IntraFormer
        self.bssfp_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.lge_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2w_encode_conv = nn.Conv2d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)

        num_tokens = patch_size ** 2  # 8x8 = 64 tokens for 128x128 image with stride 16
        self.bssfp_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))
        self.lge_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))
        self.t2w_pos = nn.Parameter(torch.zeros(1, num_tokens, transformer_basic_dims))

        self.bssfp_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.lge_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        self.t2w_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim)
        ########### IntraFormer

        ########### InterFormer
        self.multimodal_transformer = Transformer(embedding_dim=transformer_basic_dims, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv2d(transformer_basic_dims * num_modals, basic_dims * 16 * num_modals, kernel_size=1, padding=0)
        ########### InterFormer

        self.masker = MaskModal()

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls, num_modals=num_modals, basic_dims=basic_dims)
        self.decoder_sep = Decoder_sep(num_cls=num_cls, basic_dims=basic_dims)

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, mask=None):
        if isinstance(x, (list, tuple)):
            bssfp, lge, t2w = x[0], x[1], x[2]
        elif torch.is_tensor(x):
            if x.dim() == 4 and x.size(1) == 3:
                bssfp = x[:, 0:1, :, :]
                lge = x[:, 1:2, :, :]
                t2w = x[:, 2:3, :, :]
            else:
                raise ValueError(f"Expected input shape (B, 3, H, W), got {x.shape}")
        else:
            raise TypeError(f"Unexpected input type: {type(x)}")

        B = bssfp.size(0)
        if mask is None:
            mask = torch.ones((B, self.num_modals), dtype=torch.bool, device=bssfp.device)

        # 1. Extract feature pyramids from 3 hybrid encoders
        bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5 = self.bssfp_encoder(bssfp)
        lge_x1, lge_x2, lge_x3, lge_x4, lge_x5 = self.lge_encoder(lge)
        t2w_x1, t2w_x2, t2w_x3, t2w_x4, t2w_x5 = self.t2w_encoder(t2w)

        ########### IntraFormer
        bssfp_token_x5 = self.bssfp_encode_conv(bssfp_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)
        lge_token_x5 = self.lge_encode_conv(lge_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)
        t2w_token_x5 = self.t2w_encode_conv(t2w_x5).permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims)

        bssfp_intra_token_x5 = self.bssfp_transformer(bssfp_token_x5, self.bssfp_pos)
        lge_intra_token_x5 = self.lge_transformer(lge_token_x5, self.lge_pos)
        t2w_intra_token_x5 = self.t2w_transformer(t2w_token_x5, self.t2w_pos)

        bssfp_intra_x5 = bssfp_intra_token_x5.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()
        lge_intra_x5 = lge_intra_token_x5.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()
        t2w_intra_x5 = t2w_intra_token_x5.view(B, patch_size, patch_size, transformer_basic_dims).permute(0, 3, 1, 2).contiguous()

        training_mode = self.training or getattr(self, "is_training", False)
        if training_mode:
            bssfp_pred = self.decoder_sep(bssfp_x1, bssfp_x2, bssfp_x3, bssfp_x4, bssfp_x5)
            lge_pred = self.decoder_sep(lge_x1, lge_x2, lge_x3, lge_x4, lge_x5)
            t2w_pred = self.decoder_sep(t2w_x1, t2w_x2, t2w_x3, t2w_x4, t2w_x5)
        ########### IntraFormer

        # Fusion pyramids across 3 modalities
        x1 = self.masker(torch.stack((bssfp_x1, lge_x1, t2w_x1), dim=1), mask)  # B x (3*basic_dims) x H x W
        x2 = self.masker(torch.stack((bssfp_x2, lge_x2, t2w_x2), dim=1), mask)
        x3 = self.masker(torch.stack((bssfp_x3, lge_x3, t2w_x3), dim=1), mask)
        x4 = self.masker(torch.stack((bssfp_x4, lge_x4, t2w_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((bssfp_intra_x5, lge_intra_x5, t2w_intra_x5), dim=1), mask)

        ########### InterFormer
        bssfp_intra_x5_m, lge_intra_x5_m, t2w_intra_x5_m = torch.chunk(x5_intra, self.num_modals, dim=1)
        multimodal_token_x5 = torch.cat(
            (
                bssfp_intra_x5_m.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
                lge_intra_x5_m.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
                t2w_intra_x5_m.permute(0, 2, 3, 1).contiguous().view(B, -1, transformer_basic_dims),
            ),
            dim=1,
        )
        multimodal_pos = torch.cat((self.bssfp_pos, self.lge_pos, self.t2w_pos), dim=1)
        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(
            multimodal_inter_token_x5.view(
                multimodal_inter_token_x5.size(0),
                patch_size,
                patch_size,
                transformer_basic_dims * self.num_modals,
            ).permute(0, 3, 1, 2).contiguous()
        )
        x5_inter = multimodal_inter_x5

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, x5_inter)
        ########### InterFormer

        if training_mode:
            return fuse_pred, (bssfp_pred, lge_pred, t2w_pred), preds
        return fuse_pred


# Alias for canonical naming
MMFormer = Model
