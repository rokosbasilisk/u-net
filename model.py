import torch
import torch.nn as nn
from torch.nn import Conv2d as Conv2D
import torch.nn.init as init
import torch.nn.functional as F
import numpy
from torch.nn import Upsample
from math import floor,ceil

def diff_size(a,b):
    diffx = a.size()[2]-b.size()[2]
    diffy = a.size()[3]-b.size()[3]
    # if number is odd then small,small+1 
    # else equal-halves 
    return (diffx,diffy)

def odd_ceil(n):
    if (n%2==0):
        return (n//2,n//2)
    else: 
        return (floor(n/2),ceil(n/2))
class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        diff = diff_size(x1,x2)
        x2 = F.pad(x2, (odd_ceil(diff[0])[0],odd_ceil(diff[0])[1],odd_ceil(diff[1])[0],odd_ceil(diff[1]))[1],)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = F.max_pool2d(x,2)
        x = self.conv(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_conv = self.conv = nn.Sequential(
            Conv2D(1, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.up1 = Up(96, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, 1)

        
    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return x
    


    
