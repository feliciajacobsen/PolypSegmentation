import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (conv2D -> batchnorm -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.downsample_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.downsample_conv(x)


class Upsample(nn.Module):
    """
    Decoder block that takes uses a double conv block and
    bilinear upsampling. Takes also additional input, and pads original
    image to match added input.

    Args:
        in_channels (int): channel dim of input tensor
        out_channels (int): channel dim of output tensor
        x1 (tensor): original input
        x2 (tensor): skip connection input

    Returns:
        tensor
    """

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff_height = x2.shape[2] - x1.shape[2]
        diff_width = x2.shape[3] - x1.shape[3]

        # (padding_left, padding_right, padding_top, padding_bottom)
        pad_size = (
            diff_width // 2,
            diff_width - diff_width // 2,
            diff_height // 2,
            diff_height - diff_height // 2,
        )

        x1 = F.pad(x1, pad_size)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        """
        Credit : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

        Args:
            in_channels (int): iterable-style dataset.
            out_channels (int): provides with a forward method.
            features (list): list containing integers of each individual feature size of encoder network
        Returns:
            Tensor with unbounded values. Must be sigmoided to obtain probabilities.

        """

        # Downsample image
        self.input = DoubleConv(in_channels, 64)
        self.down1 = Downsample(64, 128)
        self.down2 = Downsample(128, 256)
        self.down3 = Downsample(256, 512)

        # Bottleneck
        self.bottleneck = Downsample(512, 512)

        # Upsample Image
        self.up1 = Upsample(
            1024, 256
        )  # output channel here is half as large as input channel in next block due to skip connections
        self.up2 = Upsample(512, 128)
        self.up3 = Upsample(256, 64)
        self.up4 = Upsample(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottleneck(x4)

        # add skip connections to each upsampling block
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output(x)


class UNet_dropout(nn.Module):
    def __init__(self, in_channels, out_channels, droprate):
        super(UNet_dropout, self).__init__()
        """
        Credit : https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

        Args:
            in_channels (int): iterable-style dataset.
            out_channels (int): provides with a forward method.
            features (list): list containing integers of each individual feature size of encoder network
        Returns:
            Tensor with unbounded values. Must be sigmoided to obtain probabilities.

        """

        # Downsample image
        self.input = DoubleConv(in_channels, 64)
        self.down1 = Downsample(64, 128)
        self.down2 = Downsample(128, 256)
        self.down3 = Downsample(256, 512)

        self.dropout = nn.Dropout(p=droprate)

        # Bottleneck
        self.bottleneck = Downsample(512, 512)

        # Upsample Image
        self.up1 = Upsample(
            1024, 256
        )  # output channel here is half as large as input channel in next block due to skip connections
        self.up2 = Upsample(512, 128)
        self.up3 = Upsample(256, 64)
        self.up4 = Upsample(128, 64)
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(self.dropout(x2))
        x4 = self.down3(self.dropout(x3))

        x5 = self.bottleneck(self.dropout(x4))

        # add skip connections to each upsampling block
        x = self.up1(x5, self.dropout(x4))
        x = self.up2(x, self.dropout(x3))
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output(x)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((3, 1, 160, 160)).to(device)
    model = UNet(in_channels=1, out_channels=1).to(device)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
