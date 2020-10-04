import torch 
import torch.nn as nn 
from functions import *
import torch.nn as nn 


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.input_conv = self.conv = nn.Sequential(
            nn.Conv2d(9, 8, kernel_size = 3, padding = 1),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.up1 = Up(96, 32)
        self.up2 = Up(48, 16)
        self.up3 = Up(24, 8)
        self.up4 = Squash(17,9)
        self.pool = nn.MaxPool2d((2,2),(1,1))
        self.downsize = Down(8,9)


    def forward(self, x):
        x0 = x
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x0,x1)
        return x
if __name__ == '__main__':
    import torch.nn as nn
    model = Model()
    from data import RoadMap
    data = RoadMap('./f.h5').data_array
    print(torch.Tensor(data)[1].unsqueeze(0).permute(0,3,1,2).shape)
    print(model.forward(torch.Tensor(data)[1].unsqueeze(0).permute(0,3,1,2)).shape)

