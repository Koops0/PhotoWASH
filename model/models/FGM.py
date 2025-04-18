import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, kernel_size=5):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size,padding=(kernel_size//2), bias=True)
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)  #k=1

    def forward(self, x, lrl=True):
        if lrl:
            return x + self.lff(self.layers(x))  # local residual learning
        else:
            return self.layers(x)
        
class FGM(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=5):
        super(FGM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 5,padding=(kernel_size//2), bias=True)
        self.RDB_1 = nn.Sequential(
            RDB(64, 64, 3),
            nn.ReLU(True)
        )
        
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(64,128,5,padding=(kernel_size//2), bias=True),
        #     nn.ReLU(True)
        # )
        # self.conv2 = nn.Conv2d(out_channels,in_channels, 5,padding=(kernel_size//2), bias=True)
        self.RDB_2 = nn.Sequential(
            RDB(64, 64, 3),
            nn.ReLU(True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64,64,5,padding=(kernel_size//2), bias=True),
            nn.ReLU(True)
        )

        self.conv2 = nn.Conv2d(out_channels,in_channels, 5,padding=(kernel_size//2), bias=True)

        # self.L_FGM_clip= HFRM_clip()

    def forward(self, x):
        y=x
        out1 =self.conv1(x)
        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)
        out4 = out1 + out2 + out3
        out5 = self.conv_block2(out4)
        out = self.conv2(out5)

        return y - out

