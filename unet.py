from torch import nn
import torch

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        out = None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpBlock(nn.Module):

    def __init__(
        self, in_channels, res_channels=0, last_layer=False, num_classes=None
    ) -> None:
        super(UpBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
            last_layer == True and num_classes != None
        ), "Invalid arguments"
        self.upconv1 = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(2, 2, 2),
            stride=2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(
            in_channels=in_channels + res_channels,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(
                in_channels=in_channels // 2,
                out_channels=num_classes,
                kernel_size=(1, 1, 1),
            )

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None:
            out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer:
            out = self.conv3(out)
        return out


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        level_channels=[64, 128, 256],
        bottleneck_channel=512,
    ) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = (
            level_channels[0],
            level_channels[1],
            level_channels[2],
        )
        self.down1 = DownBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.down2 = DownBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.down3 = DownBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)

        self.bottle_neck = DownBlock(
            in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True
        )

        self.up3 = UpBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.up2 = UpBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.up1 = UpBlock(
            in_channels=level_2_chnls,
            res_channels=level_1_chnls,
            num_classes=num_classes,
            last_layer=True,
        )

    def forward(self, input):
        out, residual_level1 = self.down1(input)
        out, residual_level2 = self.down2(out)
        out, residual_level3 = self.down3(out)
        out, _ = self.bottle_neck(out)

        out = self.up3(out, residual_level3)
        out = self.up2(out, residual_level2)
        out = self.up1(out, residual_level1)
        return out
