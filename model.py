import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
       super(DoubleConv, self) .__init__()
       self.conv = nn.Sequential(
           nn.Conv2D(in_channels, out_channels, 3,1,1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.RelU(inplace=True),
           nn.Conv2D(out_channels, out_channels, 3,1,1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.RelU(inplace=True)
       )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # encoder for the UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # decoder for the UNET
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2D(
                   feature*2, feature, kernel_size=2, stride=2 
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)