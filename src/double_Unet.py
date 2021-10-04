import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

def weights_init(x):
    if isinstance(x, nn.Linear):
        nn.init.kaiming_normal_(x)


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

        #self.excitation[0].apply(weights_init) # he init 
        #self.excitation[2].apply(weights_init)

    def forward(self, x):
        bs, c, _, _ = x.shape # batch size, channels, height, width
        y = self.squeeze(x).view(bs, c) # squeeze and reshape
        y = self.excitation(y).view(bs, c, 1, 1) # excitate and add h and w dimension
        return x * y.expand_as(x) # x multiplied with expanded x


class Conv2dSamePadding(nn.Conv2d):
    """
    Credit: https://www.programcreek.com/python/?code=soeaver%2FParsing-R-CNN%2FParsing-R-CNN-master%2Fmodels%2Fops%2Fconv2d_samepadding.py
    
    2D Convolutions with SAME padding like TensorFlow 
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
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


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        """
        There is no need to add bias term in the conv layer
        since batchnorm layer includes the addition of bias.
        Batchnorm layer:
        gamma * normalized(x) + bias
        """

        self.conv_block = nn.Sequential(
            Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv2dSamePadding(out_channels, out_channels, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.SE = SE_Block(out_channels)

    def forward(self, x):
        return self.SE(self.conv_block(x))



class first_encoder(nn.Module):
    def __init__(self):
        super(first_encoder, self).__init__()
        model = models.vgg19(pretrained=True) # vgg19 with 3 last FC layers
        features = list(model.features)  #[:27]
        self.features = nn.ModuleList(features).eval() 
    
    def forward(self, x):
        skip_connections = []

        # extract features from each block except last
        for ii, model in enumerate(self.features): 
            x = model(x)
            if ii in {3, 8, 17, 26}: # pre-pooled output features 
                skip_connections.append(x)
        # get output feature from vgg19
        output = self.features[-1](x)

        return output, skip_connections


class first_decoder(nn.Module):
    def __init__(self, skip_connections, features = [64, 128, 256, 512]):
        super(first_decoder, self).__init__()
        self.skip_connections = skip_connections
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.decoder.append(nn.UpsamplingBilinear2d(scale_factor=(2,2)))
            self.decoder.append(conv_block(2*feature, feature))
        
    def forward(self, x):
        skip_connections = self.skip_connections[::-1]
        step = 2 # we only want to add to the doubleconv layer

        for idx in range(0, len(self.decoder), step):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//step]
            
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            x = torch.cat((skip_connection, x), dim=1)

        return x


class second_encoder(nn.Module):
    def __init__(self, features = [64, 128, 256, 512]):
        super(second_encoder, self).__init__()
        
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=(2,2))     
    
    def forward(self, x):
        skip_connections = []
        
        for idx, feature in enumerate(self.features):
            x = conv_block(x.shape[1], feature)(x) 
            skip_connections.append(x) # pre-pooled output features 
            x = self.pool(x)
        
        return x, skip_connections


class second_decoder(nn.Module):
    def __init__(self, skip_1, skip_2, features = [64, 128, 256, 512]):
        super(second_decoder, self).__init__()
        self.skip_1 = skip_1[::-1]
        self.skip_2 = skip_2[::-1]

        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.decoder.append(nn.UpsamplingBilinear2d(scale_factor=(2,2)))
            self.decoder.append(conv_block(3*feature, feature))

    def forward(self, x):

        skip_1 = self.skip_1
        skip_2 = self.skip_2


        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)

            s_1 = skip_1[idx//2]
            s_2 = skip_2[idx//2]
           
            # funker ikke fordi skip_1 og skip_2 mangler batch dimension
            # og vil derfor ikke concattes med x pga dette
            x = torch.cat((x, s_1, s_2), dim=1)     
            
        return x
    


class output_block(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(output_block, self).__init__()
        self.block = nn.Sequential(
            Conv2dSamePadding(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.block(x)



class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        self.out_channels = out_channels

        self.relu = nn.ReLU(inplace=True)

        self.batchnorm = nn.BatchNorm2d(out_channels)

        self.conv_dilations = nn.ModuleList([])
        for dilation in [1, 6, 12, 18]:
            self.conv_dilations.append(
                nn.Sequential(
                Conv2dSamePadding(out_channels, out_channels, kernel_size=1, dilation=dilation,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
            )

    def forward(self, x):
        shape = x.shape 
        #y1 = nn.AvgPool2d(kernel_size=(shape[3], shape[1]))(x) # feil dersom denne brukes
        y1 = nn.AdaptiveAvgPool2d(1)(x)
        y1 = Conv2dSamePadding(
            in_channels=y1.shape[1], 
            out_channels=self.out_channels, 
            kernel_size=1, 
            bias=False
            )(y1)
        y1 = self.relu(self.batchnorm(y1))
        
        y1 = nn.UpsamplingBilinear2d(size=(shape[2], shape[3]))(y1) 

        y2 = self.conv_dilations[0](y1)

        y3 = self.conv_dilations[1](y2)

        y4 = self.conv_dilations[2](y3)

        y5 = self.conv_dilations[3](y4)

        y = torch.cat([y1, y2, y3, y4, y5], dim=1) # concat along channel dim
        
        y = Conv2dSamePadding(
            in_channels=(5*self.out_channels), 
            out_channels=self.out_channels, 
            kernel_size=1, 
            dilation=1, 
            bias=False
            )(y)
        y = nn.BatchNorm2d(self.out_channels)(y)
        y = F.relu(y)  

        return y


class tot_model(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(tot_model,self).__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels

    def forward(self, x):
        x, skip_1 = first_encoder()(x)
        #print("after encoder 1:", x.shape)
        x = ASPP(x.shape[1], self.out_channels)(x)
        #print("after first ASPP:", x.shape)
        x = first_decoder(skip_1)(x) 
        #print("after decoder 1:", x.shape)
        
        outputs1 = output_block(in_channels=x.shape[1])(x)
        
        x = torch.mul(x,outputs1) 
        #print("after multiplication:", x.shape)
        x, skip_2 = second_encoder()(x) 
        #print("after encoder 2:", x.shape)
        x = ASPP(self.out_channels, self.out_channels)(x)
        #print("after second ASPP:", x.shape)
        x = second_decoder(skip_1, skip_2)(x)
        #print("after decoder 2:", x.shape)  
        
        outputs2 = output_block(in_channels=x.shape[1])(x)

        outputs = torch.cat([outputs1,outputs2], dim=1)
        outputs = output_block(in_channels=2)(outputs)

        return outputs


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)) # vil ikke funke hvis batchsize=1 pga batchnorm layers
    #x = torch.randn((12,3,128, 128))
    model = tot_model(in_channels=3)
    preds = model(x)
    print(preds.shape)