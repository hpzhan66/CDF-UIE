import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers
from einops import rearrange


# Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Feedforword
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlpgated(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Mlpgated, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        if x.size(1) != self.project_in.in_channels:
            raise ValueError(f"Expected input channels {self.project_in.in_channels}, but got {x.size(1)}")
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
# Cross-Attention
class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and fourier branch are not equal!'

        b, c, h, w = x.shape

        # q = self.q(x)  # image
        # k = self.k(y)  # event
        # v = self.v(y)  # event
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Cross_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(Cross_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1_image = LayerNorm(dim, LayerNorm_type)
        self.norm1_event = LayerNorm(dim, LayerNorm_type)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        # mlp_hidden_dim = int(dim * ffn_expansion_factor)
        # self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.ffn = Mlpgated(dim, ffn_expansion_factor=2, bias=bias)

    def forward(self, image, event):
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape
        fused = image + self.attn(self.norm1_image(image), self.norm1_event(event))
        # print(fused.shape)
        # mlp
        fused = to_3d(fused)
        # print(fused.shape)
        fused_norm = self.norm2(fused)
        ffn_fused = self.ffn(to_4d(fused_norm, h, w))
        fused = to_4d(fused, h, w) + ffn_fused
        # fused = fused + self.ffn(self.norm2(fused))
        # fused = to_4d(fused, h, w)

        return fused


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)
        # import pdb
        # pdb.set_trace()

        return out


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc, nc // 2)

    def forward(self, x):
        yy = self.block(x)

        return x + yy


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class FreBlockAdjust(nn.Module):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.sft = SFT(nc)
        self.cat = nn.Conv2d(2 * nc, nc, 1, 1, 0)

    def forward(self, x, y_amp, y_phase):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(torch.cat([y_phase, pha], 1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.spatial_frequency = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class ProcessBlockAdjust(nn.Module):
    def __init__(self, in_nc, num_heads=4, bias=False):
        super(ProcessBlockAdjust, self).__init__()
        # self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockAdjust(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.spatial_frequency = nn.Conv2d(in_nc, in_nc, 3, 1, 1)
        self.cat = nn.Conv2d(2 * in_nc, in_nc, 1, 1, 0)
        self.cross_attention_block = Cross_ChannelAttentionTransformerBlock(
            dim=in_nc, num_heads=num_heads, bias=bias)

    def forward(self, x, y_amp, y_phase):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')

        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq, y_amp, y_phase)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x = self.cross_attention_block(x, x_freq_spatial)
        xcat = torch.cat([x, x_freq_spatial], 1)
        x_out = self.cat(xcat)

        return x_out + xori


class SFT(nn.Module):
    def __init__(self, nc):
        super(SFT, self).__init__()
        self.convmul = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convadd = nn.Conv2d(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2d(2 * nc, nc, 1, 1, 0)

    def forward(self, x, res):
        # res = res.detach()
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(torch.cat([x, mul * x + add], 1))
        return fuse


class HighNet(nn.Module):
    def __init__(self, nc):
        super(HighNet, self).__init__()
        self.conv0 = nn.PixelUnshuffle(2)
        self.conv1 = ProcessBlockAdjust(12)
        # self.conv2 = ProcessBlockAdjust(nc)
        self.conv3 = ProcessBlock(12)
        self.conv4 = ProcessBlock(12)
        self.conv5 = nn.PixelShuffle(2)
        self.convout = nn.Conv2d(3, 3, 3, 1, 1)
        self.trans = nn.Conv2d(6, 32, 1, 1, 0)
        self.con_temp1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.con_temp2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.con_temp3 = nn.Conv2d(32, 3, 3, 1, 1)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x, y_down, y_down_amp, y_down_phase):
        x_ori = x
        x = self.conv0(x)  # 3*4=12
        # print('pu',x.shape)

        x1 = self.conv1(x, y_down_amp, y_down_phase)
        # x2 = self.conv2(x1, y_down_amp, y_down_phase)

        x3 = self.conv3(x1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xout_temp = self.convout(x5)
        y_aff = self.trans(torch.cat([F.interpolate(y_down, scale_factor=2, mode='bilinear'), xout_temp], 1))
        con_temp1 = self.con_temp1(y_aff)
        con_temp2 = self.con_temp2(con_temp1)
        xout = self.con_temp3(con_temp2)
        # xout = coeff_apply(x_ori, y_aff)+xout

        return xout


class LowNet(nn.Module):
    def __init__(self, in_nc, nc):
        super(LowNet, self).__init__()
        self.conv0 = nn.Conv2d(in_nc, nc, 1, 1, 0)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2d(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2d(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2d(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc, 12, 1, 1, 0)
        self.convoutfinal = nn.Conv2d(12, 3, 1, 1, 0)

        self.transamp = nn.Conv2d(12, 12, 1, 1, 0)
        self.transpha = nn.Conv2d(12, 12, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(torch.cat([F.interpolate(x3, size=(x12.size()[2], x12.size()[3]), mode='bilinear'), x12], 1))
        x4 = self.conv4(x34)
        x4 = self.up2(torch.cat([F.interpolate(x4, size=(x01.size()[2], x01.size()[3]), mode='bilinear'), x01], 1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        # print('xout',xout.shape)
        xout_fre = torch.fft.rfft2(xout, norm='backward')
        xout_fre_amp, xout_fre_phase = torch.abs(xout_fre), torch.angle(xout_fre)
        xfinal = self.convoutfinal(xout)

        return xfinal, self.transamp(xout_fre_amp), self.transpha(xout_fre_phase)


class InteractNet(nn.Module):
    def __init__(self, nc=16):
        super(InteractNet, self).__init__()
        self.extract = nn.Conv2d(3, nc // 2, 1, 1, 0)
        self.lownet = LowNet(nc // 2, nc * 12)
        self.highnet = HighNet(nc)

    def forward(self, x):
        # print('x',x.shape)
        x_f = self.extract(x)
        # print('x_f', x_f.shape)
        x_f_down = F.interpolate(x_f, scale_factor=0.5, mode='bilinear')
        # print('x_f_down',x_f_down.shape)
        y_down, y_down_amp, y_down_phase = self.lownet(x_f_down)
        # print('y_down',y_down.shape)
        # print('y_down_phase',y_down_phase.shape)
        # print('y_dowm_amp',y_down_amp.shape)
        y = self.highnet(x, y_down, y_down_amp, y_down_phase)

        return y, y_down


if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512).cuda()
    # print(x.shape)
    model = InteractNet().cuda()
    output = model(x)
