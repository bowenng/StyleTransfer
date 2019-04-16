from torch import nn
from torchvision.models import vgg16
import torch


class StyleTransferNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=2)
        self.downsample_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.residual_block1 = ResidualBlock(128, 256)
        self.residual_block2 = ResidualBlock(256, 512)
        self.residual_block3 = ResidualBlock(512, 512)
        self.residual_block4 = ResidualBlock(512, 256)
        self.residual_block5 = ResidualBlock(256, 128)
        self.upsample_conv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2)
        self.upsample_conv2 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=9, stride=2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

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

    def content_loss(self, content_image, generated_image):
        out_c = self.content_loss_net(content_image).view(content_image.size()[0],-1)
        out_g = self.content_loss_net(generated_image).view(generated_image.size()[0],-1)

        CHW = out_c.size()[1]
        euclidean_distance = torch.mean(1/CHW*torch.sqrt(torch.sum((out_g - out_c)**2, dim=1)))
        return euclidean_distance


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

    def style_loss(self, style_images, generated_image):
        style_image_batch = style_images.size()[0]

        avg_loss = torch.zeros((1,))
        for style_image in style_images:
            s1, s2, s3, s4 = self(style_image.unsqueeze(0))
            g1, g2, g3, g4 = self(generated_image)

            l1 = self.frobenius_norm(s1, g1)
            l2 = self.frobenius_norm(s2, g2)
            l3 = self.frobenius_norm(s3, g3)
            l4 = self.frobenius_norm(s4, g4)
            avg_loss.add_(l1+l2+l3+l4)
        return avg_loss.div_(style_image_batch)

    def frobenius_norm(self, s_features, g_features):
        # reshape to Batch, C, H*W
        psi_s = s_features.view(s_features.size()[0], s_features.size()[1], -1)
        CHW_s = s_features.size()[1]*s_features.size()[2]*s_features.size()[3]
        G_s = torch.matmul(psi_s, psi_s.transpose(2, 1), )/CHW_s

        psi_g = g_features.view(g_features.size()[0], g_features.size()[1], -1)
        CHW_g = g_features.size()[1] * g_features.size()[2] * g_features.size()[3]
        G_g = torch.matmul(psi_g, psi_g.transpose(2, 1)) / CHW_g

        G_s = G_s.view(G_s.size()[0], -1)
        G_g = G_g.view(G_g.size()[0], -1)

        loss = torch.mean(torch.sqrt(torch.sum((G_s - G_g)**2, dim=1)))

        return loss
