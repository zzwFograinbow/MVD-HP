import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT, UnetDsv3


class AdaptiveAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(AdaptiveAttentionModule, self).__init__()
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        
        self.local_attention = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        global_att = self.global_attention(x)
        local_att = self.local_attention(x)
        
        att = global_att * local_att
        out = att * x + x 
        
        return out


class RA_UNet_3D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(RA_UNet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att1 = AdaptiveAttentionModule(filters[0])

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att2 = AdaptiveAttentionModule(filters[1])

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att3 = AdaptiveAttentionModule(filters[2])

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.att4 = AdaptiveAttentionModule(filters[3])

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], self.is_batchnorm)
        self.att5 = AdaptiveAttentionModule(filters[3])
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], self.is_batchnorm)
        self.att6 = AdaptiveAttentionModule(filters[2])
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], self.is_batchnorm)
        self.att7 = AdaptiveAttentionModule(filters[1])
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], self.is_batchnorm)
        self.att8 = AdaptiveAttentionModule(filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(filters[3], n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(filters[2], n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(filters[1], n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(filters[0], n_classes, kernel_size=1)

        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        att1 = self.att1(conv1)
        maxpool1 = self.maxpool1(att1)

        conv2 = self.conv2(maxpool1)
        att2 = self.att2(conv2)
        maxpool2 = self.maxpool2(att2)

        conv3 = self.conv3(maxpool2)
        att3 = self.att3(conv3)
        maxpool3 = self.maxpool3(att3)

        conv4 = self.conv4(maxpool3)
        att4 = self.att4(conv4)
        maxpool4 = self.maxpool4(att4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(att4, center)
        att5 = self.att5(up4)
        up4 = self.dropout1(att5)

        up3 = self.up_concat3(att3, up4)
        att6 = self.att6(up3)
        up3 = self.dropout2(att6)

        up2 = self.up_concat2(att2, up3)
        att7 = self.att7(up2)
        up2 = self.dropout3(att7)

        up1 = self.up_concat1(att1, up2)
        att8 = self.att8(up1)
        up1 = self.dropout4(att8)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        return dsv1, dsv2, dsv3, dsv4