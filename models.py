from torch import nn
from torchvision.models import vgg16
import torch
import numpy as np

class StyleTransferNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_conv1 = ConvLayer(in_channels=3, out_channels=32, kernel_size=9, stride=1)
        self.downsample_conv2 = ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.downsample_conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.residual_block1 = ResidualBlock(128, 128)
        self.residual_block2 = ResidualBlock(128, 128)
        self.residual_block3 = ResidualBlock(128, 128)
        self.residual_block4 = ResidualBlock(128, 128)
        self.residual_block5 = ResidualBlock(128, 128)
        self.upsample_conv1 = UpsampleConvLayer(in_channels=128, out_channels=64, kernel_size=3, stride=1, upsample=2)
        self.upsample_conv2 = UpsampleConvLayer(in_channels=64, out_channels=32, kernel_size=3, stride=1, upsample=2)
        self.upsample_conv3 = UpsampleConvLayer(in_channels=32, out_channels=3, kernel_size=9, stride=1)
        self.down_bn1 = InstanceNormalization(32)
        self.down_bn2 = InstanceNormalization(64)
        self.down_bn3 = InstanceNormalization(128)
        
        self.up_bn1 = InstanceNormalization(64)
        self.up_bn2 = InstanceNormalization(32)
        self.relu = nn.ReLU()
        self.tanh = ScaledTanh(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    def forward(self, x):
            initial_size = x.shape
            x = self.relu(self.down_bn1(self.downsample_conv1(x)))
            first_conv_size = x.shape
            x = self.relu(self.down_bn2(self.downsample_conv2(x)))
            x = self.relu(self.down_bn3(self.downsample_conv3(x)))
        
            x = self.residual_block1(x)
            x = self.residual_block2(x)
            x = self.residual_block3(x)
            x = self.residual_block4(x)
            x = self.residual_block5(x)

            x = self.relu(self.up_bn1(self.upsample_conv1(x)))
            x = self.relu(self.up_bn2(self.upsample_conv2(x)))
            x = self.tanh(self.upsample_conv3(x))
            return x
        
        
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class InstanceNormalization(nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
    
    
class ResidualBlock(nn.Module):
    def __init__(self,  inplanes, planes, stride=1):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = self.conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.W = None if inplanes == planes else nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1)


    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.W:
            identity = self.W(identity)

        out += identity
        out = self.relu(out)

        return out


class ContentLossNet(nn.Module):
    def __init__(self):
        super().__init__()
        # use vgg16 as content loss network
        # keep layers up to relu3_3 (activation right after conv3_3), which is 0th - 15th layer in Pytorch implementation

        self.content_loss_net = nn.Sequential(*(list(vgg16(pretrained=True).children())[0][:16]))
        for parameter in self.content_loss_net.parameters():
            parameter.requires_grad = False
        self.euclidean_distance = nn.MSELoss()

    def forward(self, c, g):
        c = self.content_loss_net(c).view(c.size()[0],-1)
        g = self.content_loss_net(g).view(g.size()[0],-1)
        CHW = c.size()[1]
        return self.euclidean_distance(c, g)/CHW
        


class StyleLossNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = list(vgg16(pretrained=True).children())[0]
        
        self.relu1_2 = vgg[:4]
        self.relu2_2 = vgg[4:9]
        self.relu3_3 = vgg[9:16]
        self.relu4_3 = vgg[16:23]
        
        self.style_loss_nets = (self.relu1_2, self.relu2_2, self.relu3_3, self.relu4_3)
        
        for loss_net in self.style_loss_nets:
            for parameter in loss_net.parameters():
                parameter.requires_grad = False
        
        self.mse = nn.MSELoss()
        
    def forward(self, s, g):
        loss = 0
        for loss_net in self.style_loss_nets:
            s = loss_net(s)
            g = loss_net(g)
            loss = loss + self.frobenius_norm(s, g)
        loss = loss/len(self.style_loss_nets)    
        return loss
  
    def gram_matrix(self, x):
        x = x.view(x.size()[0], x.size()[1], -1)
        CHW = x.size()[1]*x.size()[2]
        
        x = torch.matmul(x, x.transpose(2, 1)) /CHW
        return x
        
    def frobenius_norm(self, s, g):
        # reshape to Batch, C, H*W
        s = self.gram_matrix(s)
        g = self.gram_matrix(g)

        s = s.view(s.size()[0], -1)
        g = g.view(g.size()[0], -1)

        return self.mse(s, g)

    
class FeatureStyleLoss(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.style_loss_net = StyleLossNet()
        self.content_loss_net = ContentLossNet()
        self.c = c
        self.s = s
        
    def forward(self, content, style, generated):
        content_loss = self.content_loss_net(content, generated)
        style_loss = self.style_loss_net(style, generated)
        return self.c * self.content_loss_net(content, generated) \
                   + self.s * style_loss
        

class ScaledTanh(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.FloatTensor(mean).view(1,3,1,1)
        std = torch.FloatTensor(std).view(1,3,1,1)
        self.scale = (1/2/std)-(1e-5)
        max_old = (1-mean)/std
        self.bias = max_old-self.scale
        self.tanh = nn.Tanh()
        self.register_buffer('k', self.scale)
        self.register_buffer('b', self.bias)
        
    def forward(self, x):
        x = self.tanh(x)
        x = self.k * x + self.b
        return x
    
    