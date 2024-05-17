import torch
import torch.nn as nn
import math

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv_BN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )
        self.conv_GN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.GroupNorm(2, out_channels, eps=1e-05, affine=True, device=None, dtype=None),
            nn.ReLU(True)
        )

    def forward(self, x):
        if x.shape[0] == 1:
            return self.conv_GN(x)
        else:
            return self.conv_BN(x) 

class _ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(_ConvBNSig, self).__init__()
        self.conv_BN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # TODO gpt說多gpu用sync效果更不錯（待測試）
            # nn.SyncBatchNorm(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True)
        )
        self.conv_GN = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.GroupNorm(2, out_channels, eps=1e-05, affine=True, device=None, dtype=None),
            nn.ReLU(True)
        )

    def forward(self, x):
        if x.shape[0] == 1:
            return self.conv_GN(x)
        else:
            return self.conv_BN(x) 


class lwa(nn.Module):
    def __init__(self,channel,outsize):
        super().__init__()
        #self.rgbd = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.dept = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)
        self.rgb  = nn.Conv2d(channel, channel, kernel_size=(1, 1), bias=False)

        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        self.mlp = nn.Sequential(_ConvBNReLU(channel, 24, 1, 1),_ConvBNSig(24,outsize,1,1))

    
    def forward(self,rgb,dep):

        assert rgb.size() == dep.size()

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # rgb = rgb.to(device)
        # dep = dep.to(device)
        rgbd = rgb+dep
        m_batchsize,C,width ,height = rgb.size()

        proj_rgb  = self.rgb(rgb).view(m_batchsize,-1,height*width).permute(0,2,1) # B X (H*W) X C
        proj_dep  = self.dept(dep).view(m_batchsize,-1,height*width) # B X C x (H*W)
        energy    = torch.bmm(proj_rgb,proj_dep)/math.sqrt(C)  #B X (H*W) X (H*W)
        attention1 = self.softmax1(energy) #B X (H*W) X (H*W) 


        att_r = torch.bmm(proj_rgb.permute(0,2,1),attention1)
        att_b = torch.bmm(proj_dep,attention1 )
        #proj_rgbd = self.rgbd(rgbd).view(m_batchsize,-1,height*width) # B X C X (H*W) 
        #attention2 = torch.bmm(proj_rgbd,attention1.permute(0,2,1) )
        attention2 = att_r + att_b
        output = attention2.view(m_batchsize,C,width,height) + rgbd

        # GapOut = self.GAP(output)

        # training 和val分開
        if output.size(2) > 1 and output.size(3) > 1:
            GapOut = self.GAP(output)
        else:
            GapOut = output.mean(dim=(2, 3))  # 在通道维度上计算平均值

        gate = self.mlp(GapOut)

        return gate