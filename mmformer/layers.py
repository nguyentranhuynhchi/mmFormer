import torch
import torch.nn as nn


def normalization(planes, norm='in'):
    if norm == 'bn':
        m = nn.BatchNorm2d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm2d(planes)
    elif norm == 'sync_bn':
        m = nn.SyncBatchNorm(planes)
    else:
        raise ValueError(f'normalization type {norm} is not supported')
    return m


class general_conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)

    def forward(self, x):
        return self.conv(x)


class general_conv2d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv2d_prenorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(in_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


class general_conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True, act_type='lrelu', relufactor=0.2):
        super(general_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# Aliases for 3D naming compatibility
general_conv3d_prenorm = general_conv2d_prenorm
general_conv3d = general_conv2d


class prm_generator_laststage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator_laststage, self).__init__()
        self.embedding_layer = nn.Sequential(
            general_conv2d(in_channel * 4, int(in_channel // 4), k_size=1, padding=0, stride=1),
            general_conv2d(int(in_channel // 4), int(in_channel // 4), k_size=3, padding=1, stride=1),
            general_conv2d(int(in_channel // 4), in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            general_conv2d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv2d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.prm_layer(self.embedding_layer(x))


class prm_generator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4):
        super(prm_generator, self).__init__()
        self.embedding_layer = nn.Sequential(
            general_conv2d(in_channel * 4, int(in_channel // 4), k_size=1, padding=0, stride=1),
            general_conv2d(int(in_channel // 4), int(in_channel // 4), k_size=3, padding=1, stride=1),
            general_conv2d(int(in_channel // 4), in_channel, k_size=1, padding=0, stride=1),
        )
        self.prm_layer = nn.Sequential(
            general_conv2d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv2d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x1, x2):
        return self.prm_layer(torch.cat((x1, self.embedding_layer(x2)), dim=1))


class modal_fusion(nn.Module):
    def __init__(self, in_channel=64, num_modals=3):
        super(modal_fusion, self).__init__()
        self.num_modals = num_modals
        self.weight_layer = nn.Sequential(
            nn.Conv2d(num_modals * in_channel + 1, 128, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, num_modals, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm, region_name=""):
        B, K, C, H, W = x.size()
        prm_avg = torch.mean(prm, dim=(3, 4), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4), keepdim=False) / prm_avg
        feat_avg = feat_avg.view(B, K * C, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1)
        return torch.sum(x * weight, dim=1)


class region_fusion_laststage(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion_laststage, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv2d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            general_conv2d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv2d(in_channel, in_channel, k_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        B, _, _, H, W = x.size()
        x = torch.reshape(x, (B, -1, H, W))
        return self.fusion_layer(x)


class region_fusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=4):
        super(region_fusion, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv2d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            general_conv2d(in_channel, in_channel, k_size=3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.fusion_layer(x)


class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=3, num_modals=None):
        super(fusion_prenorm, self).__init__()
        modals = num_modals if num_modals is not None else num_cls
        self.fusion_layer = nn.Sequential(
            general_conv2d_prenorm(in_channel * modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv2d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv2d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        return self.fusion_layer(x)


class region_aware_modal_fusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=4, num_modals=3):
        super(region_aware_modal_fusion, self).__init__()
        self.num_cls = num_cls
        self.num_modals = num_modals

        self.modal_fusion = nn.ModuleList([modal_fusion(in_channel=in_channel, num_modals=num_modals) for _ in range(num_cls)])
        self.region_fusion = region_fusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            general_conv2d(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv2d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv2d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1),
        )

    def forward(self, x, prm):
        B, _, H, W = x.size()
        y = x.view(B, self.num_modals, -1, H, W)
        B, K, C, H, W = y.size()

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1)
        modal_feat = y.unsqueeze(2) * prm.unsqueeze(1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        region_fused_feat = []
        for i in range(self.num_cls):
            region_fused_feat.append(self.modal_fusion[i](region_feat[i], prm[:, i:i + 1, ...]))
        region_fused_feat = torch.stack(region_fused_feat, dim=1)

        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W))), dim=1)
        return final_feat
