from torch import nn
from torchvision.models import vgg16
import torch


class StyleTransferNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=2)
        self.downsample_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.residual_block1 = ResidualBlock(128, 256)
        self.residual_block2 = ResidualBlock(256, 256)
        self.residual_block3 = ResidualBlock(256, 256)
        self.residual_block4 = ResidualBlock(256, 256)
        self.residual_block5 = ResidualBlock(256, 128)
        self.upsample_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.upsample_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, stride=2)
        self.relu = nn.ReLU()
        self.tanh = ScaledTanh(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def forward(self, x):
        initial_size = x.shape
        x = self.relu(self.downsample_conv1(x))
        first_conv_size = x.shape
        x = self.relu(self.downsample_conv2(x))

        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)

        x = self.relu(self.upsample_conv1(x, output_size=first_conv_size))
        x = self.tanh(self.upsample_conv2(x, output_size=initial_size))
        return x


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

    def forward(self, x):
        return self.content_loss_net(x)

    def content_loss(self, c, g):
        c = self.content_loss_net(c).view(c.size()[0],-1)
        g = self.content_loss_net(g).view(g.size()[0],-1)

        CHW = c.size()[1]
        return torch.mean(1/CHW*torch.sqrt(torch.sum((g - c)**2, dim=1)))


class StyleLossNet(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = list(vgg16(pretrained=True).children())[0]
        self.relu1_2 = vgg[:4]
        self.relu2_2 = vgg[4:9]
        self.relu3_3 = vgg[9:16]
        self.relu4_3 = vgg[16:23]

    def forward(self, x):
        feature1 = self.relu1_2(x)
        feature2 = self.relu2_2(feature1)
        feature3 = self.relu3_3(feature2)
        feature4 = self.relu4_3(feature3)

        return (feature1, feature2, feature3, feature4)

    def style_loss(self, style_image, generated_image):
        s1, s2, s3, s4 = self(style_image)
        g1, g2, g3, g4 = self(generated_image)
        
        return (self.frobenius_norm(s1, g1)+self.frobenius_norm(s2, g2)+self.frobenius_norm(s3, g3)+self.frobenius_norm(s4, g4))
        

    def frobenius_norm(self, s, g):
        # reshape to Batch, C, H*W
        s = s.view(s.size()[0], s.size()[1], -1)
        CHW_s = s.size()[0]*s.size()[1]*s.size()[2]
        s = torch.matmul(s, s.transpose(2, 1), )/CHW_s

        g = g.view(g.size()[0], g.size()[1], -1)
        CHW_g = g.size()[0] * g.size()[1] * g.size()[2]
        g = torch.matmul(g, g.transpose(2, 1)) / CHW_g

        s = s.view(s.size()[0], -1)
        g = g.view(g.size()[0], -1)

        return torch.mean(torch.sqrt(torch.sum((s - g)**2, dim=1)))

        

    
class FeatureStyleLoss(nn.Module):
    def __init__(self, c, s):
        super().__init__()
        self.style_loss_net = StyleLossNet()
        self.content_loss_net = ContentLossNet()
        self.c = c
        self.s = s
        
    def forward(self, content, style, generated):
        return self.c * self.content_loss_net.content_loss(content, generated) \
                   + self.s * self.style_loss_net.style_loss(style, generated)
        

class ScaledTanh(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.FloatTensor(mean).view(1,3,1,1)
        std = torch.FloatTensor(std).view(1,3,1,1)
        self.scale = (1/2/std)
        max_old = (1-mean)/std
        self.bias = max_old-self.scale
        self.tanh = nn.Tanh()
        self.register_buffer('k', self.scale)
        self.register_buffer('b', self.bias)
        
    def forward(self, x):
        x = self.tanh(x)
        x = self.k * x + self.b
        return x
    
    