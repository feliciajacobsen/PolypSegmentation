import torch
import torch.nn as nn
from utils.modules import ASPP, SE_Block, Res_Conv, Attention_Block


class ResUnetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()
        """
        credit: https://github.com/rishikksh20/ResUnet/blob/master/core/res_unet_plus.py

        Resunetplusplus model
        """

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = SE_Block(filters[0])

        self.residual_conv1 = Res_Conv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = SE_Block(filters[1])

        self.residual_conv2 = Res_Conv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = SE_Block(filters[2])

        self.residual_conv3 = Res_Conv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = Attention_Block(filters[2], filters[4], filters[4])
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv1 = Res_Conv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = Attention_Block(filters[1], filters[3], filters[3])
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv2 = Res_Conv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = Attention_Block(filters[0], filters[2], filters[2])
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv3 = Res_Conv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out



class ResUnetPlusPlus_dropout(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.3, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus_dropout, self).__init__()
        """
        Resunetplusplus model
        """

        self.dropout = nn.Dropout(p=droprate)

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = SE_Block(filters[0])

        self.residual_conv1 = Res_Conv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = SE_Block(filters[1])

        self.residual_conv2 = Res_Conv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = SE_Block(filters[2])

        self.residual_conv3 = Res_Conv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = Attention_Block(filters[2], filters[4], filters[4])
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv1 = Res_Conv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = Attention_Block(filters[1], filters[3], filters[3])
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv2 = Res_Conv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = Attention_Block(filters[0], filters[2], filters[2])
        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_residual_conv3 = Res_Conv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(self.dropout(x3))

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(self.dropout(x4))

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(self.dropout(x7))

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(self.dropout(x8))

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out


def test_resunetplusplus():
    x = torch.randn((2, 3, 256, 256))
    model = ResUnetPlusPlus(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


def test_resunetplusplus_dropout():
    x = torch.randn((2, 3, 256, 256))
    model = ResUnetPlusPlus(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)


if __name__ == "__main__":
    test_resunetplusplus()
    test_resunetplusplus_dropout()
