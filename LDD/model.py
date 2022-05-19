# gdc_model.py
from email import header
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, class_num):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding
                                 , bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=False)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=False)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=False)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=False)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=class_num, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class EffUNet(nn.Module):
    """ U-Net with EfficientNet-B0 encoder """

    def __init__(self, in_channels, classes, pre_model_no_fc):
        super().__init__()
        self.conv1 = nn.Conv2d(112, 320, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(40, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(24, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)

        self.encoder = nn.Sequential(*pre_model_no_fc)
        self.down_block_1 = nn.Sequential(*pre_model_no_fc[0])
        self.down_block_2 = nn.Sequential(*pre_model_no_fc[1:3]) 
        self.down_block_3 = nn.Sequential(*pre_model_no_fc[3])
        self.down_block_4 = nn.Sequential(*pre_model_no_fc[4:6])
        self.down_block_5 = nn.Sequential(*pre_model_no_fc[6:8])
        li = [self.down_block_1, self.down_block_2, self.down_block_3, self.down_block_4, self.down_block_5]

        for block in li:
          for param in block.parameters(): 
            param.requires_grad = False

        self.up_block_4 = DecoderBlock(640, 256)
        self.up_block_3 = DecoderBlock(512, 128)
        self.up_block_2 = DecoderBlock(256, 64)
        self.up_block_1a = DecoderBlock(128, 32)
        self.up_block_1b = DecoderBlock(32, 16)
        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=1, bias=False)

    def forward(self, x):
        # cam original shape -> (batch x 1 x 224 x 224)

        # endcoder
        x1 = self.down_block_1(x)
        x2 = self.down_block_2(x1)
        x3 = self.down_block_3(x2)
        x4 = self.down_block_4(x3)
        x5 = self.down_block_5(x4)

        # decoder
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv1(x4)], dim=1)
        x5 = self.up_block_4(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv2(x3)], dim=1)
        x5 = self.up_block_3(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv3(x2)], dim=1)
        x5 = self.up_block_2(x5)

        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv4(x1)], dim=1)
        x5 = self.up_block_1a(x5)
        
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output

class EffUNet_B7(nn.Module):
    """ U-Net with EfficientNet-B7 encoder """

    def __init__(self, in_channels, classes, pre_model_no_fc):
        super().__init__()
        self.encoder = nn.Sequential(*pre_model_no_fc)
        self.down_block_1 = nn.Sequential(*pre_model_no_fc[0])
        self.down_block_2 = nn.Sequential(*pre_model_no_fc[1:3]) 
        self.down_block_3 = nn.Sequential(*pre_model_no_fc[3])
        self.down_block_4 = nn.Sequential(*pre_model_no_fc[4:6])
        self.down_block_5 = nn.Sequential(*pre_model_no_fc[6:8])
        li = [self.down_block_1, self.down_block_2, self.down_block_3, self.down_block_4, self.down_block_5]

        for block in li:
          for param in block.parameters(): 
            param.requires_grad = False

        self.conv1 = nn.Conv2d(224, 640, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(80, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(48, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.up_block_4 = DecoderBlock(1280, 256)
        self.up_block_3 = DecoderBlock(512, 128)
        self.up_block_2 = DecoderBlock(256, 64)
        self.up_block_1a = DecoderBlock(128, 32)
        self.up_block_1b = DecoderBlock(32, 16)
        self.head_conv = nn.Conv2d(in_channels=16, out_channels=classes, kernel_size=1, bias=False)

    def forward(self, x):
        # endcoder
        x1 = self.down_block_1(x) # 64
        x2 = self.down_block_2(x1) # 48
        x3 = self.down_block_3(x2) # 80
        x4 = self.down_block_4(x3) # 224
        x5 = self.down_block_5(x4) # 640

        # decoder
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv1(x4)], dim=1)
        x5 = self.up_block_4(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv2(x3)], dim=1)
        x5 = self.up_block_3(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv3(x2)], dim=1)
        x5 = self.up_block_2(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = torch.cat([x5, self.conv4(x1)], dim=1)
        x5 = self.up_block_1a(x5)
        x5 = F.interpolate(x5, scale_factor=2)
        x5 = self.up_block_1b(x5)
        output = self.head_conv(x5)

        return output

# if __name__ == '__main__':
    