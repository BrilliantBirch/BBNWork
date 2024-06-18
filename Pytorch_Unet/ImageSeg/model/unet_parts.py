"""UNet模型模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    双层卷积模块 (卷积=》批归一化=>ReLU)*2
    """
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.double_conv(x)
        
class Down(nn.Module):
    """下采样模块"""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #最大池化
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """上采样模块"""
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        #特征融合 如果是双线性插值则采取双线性插值，否则使用反卷积
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        #input is CHW -channel Height Width
        #高度上的差异
        diffY = torch.tensor([x2.size()[2]-x1.size()[2]])
        #宽度上的差异
        diffX = torch.tensor([x2.size()[3]-x1.size()[3]])
        #填充 [左，右，上，下]
        x1 = F.pad(x1,[diffX//2,diffX-diffX//2,
                       diffY//2,diffY-diffY//2])
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)
        
class OutConv(nn.Module):
    """输出模块"""
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        
    def forward(self,x):
        return self.conv(x)
    