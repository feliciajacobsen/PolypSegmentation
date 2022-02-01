import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np

from modules import SE_Block, ASPP, Res_Shortcut



class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_enc=True, residual=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SE_Block(out_channels, 8)
        )


        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shortcut = Res_Shortcut(in_channels, out_channels)

        self.is_enc = is_enc
        self.residual = residual

    def forward(self, x):
        y = self.conv(x)
 
        if self.residual:
            y = y + self.shortcut(x)

        if self.is_enc:
            y = self.pool(y)

        return y


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        

        self.aspp_blocks = nn.ModuleList()
        for rate in [6, 12, 18]:
            self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ))

        self.output = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        y_pool0 = nn.AdaptiveAvgPool2d(output_size=1)(x)
        y_conv0 = self.conv_block(y_pool0)
        y_conv0 = nn.Upsample(size=(h, w), mode='bilinear')(y_conv0)
        y_conv1 = self.conv_block(x)
        y_conv2 = self.aspp_blocks[0](x)
        y_conv3 = self.aspp_blocks[1](x)
        y_conv4 = self.aspp_blocks[2](x)

        y = torch.cat([y_conv0, y_conv1, y_conv2, y_conv3, y_conv4], 1)
        y = self.output(y)
        y = self.relu(y)
        y = self.bn(y)

        return y


def output_block():
    return nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))


class DoubleUNet(nn.Module):
    """
    Credit: https://github.com/HotVar/Image-segmentation-DoubleUNet/blob/main/model.py
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.VGG = nn.Sequential(
            VGGBlock(in_channels, 64, True),
            VGGBlock(64, 128, True),
            VGGBlock(128, 256, True),
            VGGBlock(256, 512, True),
            VGGBlock(512, 512, True)
        )

        # apply pretrained vgg19 weights on 1st unet
        vgg19 = models.vgg19_bn()
        #vgg19.load_state_dict(torch.load(PATH_VGG19))

        for i, j in zip(range(5), np.array([0,7,14,27,33])):
            self.VGG[i].conv[0].weights = vgg19.features[0+j].weight
            self.VGG[i].conv[1].weights = vgg19.features[1+j].weight
            self.VGG[i].conv[3].weights = vgg19.features[3+j].weight
            self.VGG[i].conv[4].weights = vgg19.features[4+j].weight
        del vgg19

        self.aspp1 = ASPP(512, 512)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.decoder1 = nn.Sequential(
            VGGBlock(1024, 256, False), 
            VGGBlock(512, 128, False),
            VGGBlock(256, 64, False),
            VGGBlock(128, 32, False)
        )

        self.output1 = output_block()

        self.encoder2 = nn.Sequential(
            VGGBlock(3, 64, True, True),
            VGGBlock(64, 128, True, True),
            VGGBlock(128, 256, True, True),
            VGGBlock(256, 512, True, True),
            VGGBlock(512, 512, True, True)
        )

        self.aspp2 = ASPP(512, 512)

        self.decoder2 = nn.Sequential(
            VGGBlock(1536, 256, False, True),
            VGGBlock(768, 128, False, True),
            VGGBlock(384, 64, False, True),
            VGGBlock(192, 32, False, True)
        )

        self.output2 = output_block()

        self.output = nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, _input):
        # encoder of 1st unet
        y_enc1_1 = self.VGG[0](_input)
        y_enc1_2 = self.VGG[1](y_enc1_1)
        y_enc1_3 = self.VGG[2](y_enc1_2)
        y_enc1_4 = self.VGG[3](y_enc1_3)
        y_enc1_5 = self.VGG[4](y_enc1_4)

        # aspp bridge1
        y_aspp1 = self.aspp1(y_enc1_5)

        # decoder of 1st unet
        y_dec1_4 = self.up(y_aspp1)
        y_dec1_4 = self.decoder1[0](torch.cat([y_enc1_4, y_dec1_4], 1))
        y_dec1_3 = self.up(y_dec1_4)
        y_dec1_3 = self.decoder1[1](torch.cat([y_enc1_3, y_dec1_3], 1))
        y_dec1_2 = self.up(y_dec1_3)
        y_dec1_2 = self.decoder1[2](torch.cat([y_enc1_2, y_dec1_2], 1))
        y_dec1_1 = self.up(y_dec1_2)
        y_dec1_1 = self.decoder1[3](torch.cat([y_enc1_1, y_dec1_1], 1))
        y_dec1_0 = self.up(y_dec1_1)

        # output of 1st unet
        output1 = self.output1(y_dec1_0)

        # multiply input and output of 1st unet
        mul_output1 = _input * output1

        # encoder of 2nd unet
        y_enc2_1 = self.encoder2[0](mul_output1)
        y_enc2_2 = self.encoder2[1](y_enc2_1)
        y_enc2_3 = self.encoder2[2](y_enc2_2)
        y_enc2_4 = self.encoder2[3](y_enc2_3)
        y_enc2_5 = self.encoder2[4](y_enc2_4)

        # aspp bridge 2
        y_aspp2 = self.aspp2(y_enc2_5)

        # decoder of 2nd unet
        y_dec2_4 = self.up(y_aspp2)
        y_dec2_4 = self.decoder2[0](torch.cat([y_enc1_4, y_enc2_4, y_dec2_4], 1))
        y_dec2_3 = self.up(y_dec2_4)
        y_dec2_3 = self.decoder2[1](torch.cat([y_enc1_3, y_enc2_3, y_dec2_3], 1))
        y_dec2_2 = self.up(y_dec2_3)
        y_dec2_2 = self.decoder2[2](torch.cat([y_enc1_2, y_enc2_2, y_dec2_2], 1))
        y_dec2_1 = self.up(y_dec2_2)
        y_dec2_1 = self.decoder2[3](torch.cat([y_enc1_1, y_enc2_1, y_dec2_1], 1))
        y_dec2_0 = self.up(y_dec2_1)

        # output of 2nd unet
        output2 = self.output2(y_dec2_0)

        outputs = torch.cat([output1,output2], dim=1)

        return self.output(outputs)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((3, 1, 160, 160)).to(device)
    model = DoubleUNet(in_channels=1, out_channels=1).to(device)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)) 
    model = DoubleUNet(in_channels=3, out_channels=1)
    preds = model(x)
    print(preds.shape)
   