from torch import nn


class StyleTransferNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=2)
        self.downsample_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.residual_block1 = ResidualBlock(128, 128)
        self.residual_block2 = ResidualBlock(128, 128)
        self.residual_block3 = ResidualBlock(128, 128)
        self.residual_block4 = ResidualBlock(128, 128)
        self.residual_block5 = ResidualBlock(128, 128)
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

        out += identity
        out = self.relu(out)

        return out
