import torch 
import torch.nn as nn 
import torch.functional as F 

class Squash(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Squash,self).__init__()
        #self.upsample = nn.Upsample(scale_factor=(2,2),mode='bilinear')
        self.relu = nn.ReLU()
        #self.batchnorm = nn.BatchNorm2d(10)
        self.conv = nn.Conv2d(channel_in,channel_out,kernel_size=(3,3))
    def forward(self,x1,x2):
        #x1 = self.upsample(x1)
        x_t = torch.zeros((x2.shape[0],x2.shape[1],x1.shape[2],x1.shape[3]))
        x_t[:,:,:x1.shape[2],:x1.shape[3]] = x2[:,:,:x1.shape[2],:x1.shape[3]]
        x = torch.cat([x_t,x1],dim=1)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Up(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Up,self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2,2),mode='bilinear')
        self.relu = nn.ReLU()
        #self.batchnorm = nn.BatchNorm2d(10)
        self.conv = nn.Conv2d(channel_in,channel_out,kernel_size=(3,3))
    def forward(self,x1,x2):
        x1 = self.upsample(x1)
        x_t = torch.zeros((x2.shape[0],x2.shape[1],x1.shape[2],x1.shape[3]))
        x_t[:,:,:x1.shape[2],:x1.shape[3]] = x2[:,:,:x1.shape[2],:x1.shape[3]]
        x = torch.cat([x_t,x1],dim=1)
        x = self.conv(x)
        #x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Down(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Down,self).__init__()
        self.pool = nn.MaxPool2d((2,2),(2,2))
        self.relu = nn.ReLU()
        #self.batchnorm = nn.BatchNorm2d(10)
        self.conv = nn.Conv2d(channel_in,channel_out,kernel_size=(3,3))
    def forward(self,x):
        return self.relu(self.pool(self.conv(x)))

if __name__ == '__main__':
    a = torch.randn(10,10,10,10)
    b = torch.randn(10,10,10,10)
    up = Up(10,10)
    b = up(a,b)
    print(b.shape)
    down = Down(10,10)
    print(down(a).shape)

