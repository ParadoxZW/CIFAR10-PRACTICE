'''
implement some basic module and blocks for furthor building
'''
from torch import nn
import torch


def Conv2d(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding)
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(conv, bn, nn.ReLU())


class FMPBlock(nn.Module):
    "Block with fractional max pooling"

    def __init__(self, in_channels, out_channels, output_ratio=0.8):
        super(FMPBlock, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=3, stride=1, padding=1)
        # self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.fmp = nn.FractionalMaxPool2d(2, output_ratio=output_ratio)

    def forward(self, x):
        return self.fmp(self.c1(x))


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).double().cuda()
        self.b_2 = nn.Parameter(torch.zeros(features)).double().cuda()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Self_Attn(nn.Module):
    "Self-attention Layer for cnn"

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Connection(nn.Module):
    """
    A residual connection followed by a batch normalization.
    """

    def __init__(self, channel):
        super(Connection, self).__init__()
        # self.norm = nn.BatchNorm2d(channel)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + sublayer(x)


class DotConnection(nn.Module):
    """
    A residual connection when dimention increases.
    """

    def __init__(self, width, channel):
        super(DotConnection, self).__init__()
        # self.norm = nn.BatchNorm2d(channel)
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.zeros = torch.zeros(
            (64,  channel, int(width / 2), int(width / 2))).double().cuda()  # set 128 channels if only one gpu

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the increase size."
        y = self.pooling(x)
        y = torch.cat((y, self.zeros), 1)
        return y + sublayer(x)


class ResBlock(nn.Module):
    """
    A block for building resnet.
    """

    def __init__(self, channels, kernel_size=3):
        super(ResBlock, self).__init__()
        c1 = Conv2d(channels, channels,
                    kernel_size=kernel_size, stride=1, padding=1)
        c2 = Conv2d(channels, channels,
                    kernel_size=kernel_size, stride=1, padding=1)
        self.conv = nn.Sequential(c1, c2)
        self.shortcut = Connection(channels)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return self.shortcut(x, self.conv)


class SampleResBlock(nn.Module):
    """
    A block for building resnet.
    """

    def __init__(self, in_channels, in_width, kernel_size=3):
        super(SampleResBlock, self).__init__()
        c1 = Conv2d(in_channels, in_channels * 2,
                    kernel_size=kernel_size, stride=2, padding=1)
        c2 = Conv2d(in_channels * 2, in_channels * 2,
                    kernel_size=kernel_size, stride=1, padding=1)
        self.conv = nn.Sequential(c1, c2)
        self.shortcut = DotConnection(width=in_width, channel=in_channels)

    def forward(self, x):
        "Apply residual connection to any sublayer with the same size."
        return self.shortcut(x, self.conv)
