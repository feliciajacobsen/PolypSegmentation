import torch
import torch.nn as nn



class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, filters, r=8):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # squeeze each image to a single value for each channel
        self.excitation = nn.Sequential(
            nn.Linear(filters, filters // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(filters // r, filters, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape # batch size, channels, height, width
        y = self.squeeze(x).view(bs, c) # squeeze and reshape
        y = self.excitation(y).view(bs, c, 1, 1) # excitate and add h and w dimension
        return x * y.expand_as(x) # x multiplied with expanded x


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Conv2dSamePadding(nn.Conv2d):
    """
    Credit: https://www.programcreek.com/python/?code=soeaver%2FParsing-R-CNN%2FParsing-R-CNN-master%2Fmodels%2Fops%2Fconv2d_samepadding.py
    
    2D Convolutions with SAME padding like TensorFlow 
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv2dSamePadding, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_blocks = nn.ModuleList()
        for rate in rates:
            self.aspp_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=rate, dilation=rate),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        ))
            
        self.output = nn.Conv2d(len(rates) * out_channels, out_channels, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_blocks[0](x)
        x2 = self.aspp_blocks[1](x)
        x3 = self.aspp_blocks[2](x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



class Res_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(Res_Conv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)



class Attention_Block(nn.Module):
    def __init__(self, input_encoder, input_decoder, out_channels):
        super(Attention_Block, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_encoder, out_channels, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_decoder, out_channels, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2
