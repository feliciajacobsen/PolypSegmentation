import torch
import torch.nn as nn
import torchvision.transforms.functional as TF



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNET, self).__init__()

        """
        Function perform one epoch on entire dataset and outputs loss for each batch.

        Args:
            in_channels (int): iterable-style dataset.

            out_channels (int): provides with a forward method.

            features (list): list containing integers of each individual feature size of encoder network

        Returns:
            Predicted .jpg-image of segmentation mask.
        """

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsample image
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # Upsample image
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    2*feature, feature, kernel_size=2, stride=2
                )
            )    
            self.ups.append(DoubleConv(2*feature, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # downsampling part of unet
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # reverse list
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)



def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((3, 1, 160, 160)).to(device)
    model = UNET(in_channels=1, out_channels=1, features=[64,128,256,512]).to(device)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape



if __name__ == "__main__":
    test()