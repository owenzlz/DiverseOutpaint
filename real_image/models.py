import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb
from torchvision.models import resnet18
from spectral_normalization import SpectralNorm

##############################
#        Conv Layers
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(out_size, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

##############################
#        Encoder
##############################

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)
        self.down7 = UNetDown(512, 512, normalize=False)        

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d7 = d7.view(d7.shape[0], -1)
        return d7, [d1, d2, d3, d4, d5, d6]

##############################
#        Decoder
##############################

class Decoder(nn.Module):
    def __init__(self, noise_dim):
        super(Decoder, self).__init__()
        self.up1 = UNetUp(512+noise_dim, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)
        
        final = [   nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
                    nn.Tanh() ]
        self.final = nn.Sequential(*final)     

    def forward(self, c_z, feats):
        [d1, d2, d3, d4, d5, d6] = feats
        u1 = self.up1(c_z, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)

##############################
#        Discriminator
##############################

class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(4):
            self.models.add_module('disc_%d' % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                )
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)
        
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt)**2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.conv6 = SpectralNorm(nn.Conv2d(1024, 1024, 4, 1, 0))
        self.conv7 = SpectralNorm(nn.Conv2d(1024, 1, 1, 1, 0))

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        z = self.conv7(x)
        return z

class PSPDiscriminator(nn.Module):
    def __init__(self, d=128):
        super(PSPDiscriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1))

        # pyramid conv
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_16_1 = SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1))
        self.conv_16_2 = SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1))
        self.conv_8_1 = SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1))
        self.conv_8_2 = SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1))
        self.conv_4_1 = SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1))
        self.conv_4_2 = SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1))
        self.conv_2_1 = SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1))
        self.conv_2_2 = SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1))

        # final conv
        self.conv3 = SpectralNorm(nn.Conv2d(256, 256, 4, 2, 1))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, 2, 1))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 1024, 4, 2, 1))
        self.conv6 = SpectralNorm(nn.Conv2d(1024, 1024, 4, 1, 0))
        self.conv7 = SpectralNorm(nn.Conv2d(1024, 1, 1, 1, 0))

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        
        # pool features into smaller spatial dimensions
        down_16 = self.pool(x)
        down_8 = self.pool(down_16)
        down_4 = self.pool(down_8)
        down_2 = self.pool(down_4)

        # conv features into lower dimensional channels
        down_16 = F.leaky_relu(self.conv_16_1(down_16), 0.2)
        down_16 = F.leaky_relu(self.conv_16_2(down_16), 0.2)
        down_8 = F.leaky_relu(self.conv_8_1(down_8), 0.2)
        down_8 = F.leaky_relu(self.conv_8_2(down_8), 0.2)
        down_4 = F.leaky_relu(self.conv_4_1(down_4), 0.2)
        down_4 = F.leaky_relu(self.conv_4_2(down_4), 0.2)
        down_2 = F.leaky_relu(self.conv_2_1(down_2), 0.2)
        down_2 = F.leaky_relu(self.conv_2_2(down_2), 0.2)
        
        # upsample features and concatenate pyramid features
        down_16 = F.upsample(down_16, (32,32))
        down_8 = F.upsample(down_8, (32,32))
        down_4 = F.upsample(down_4, (32,32))
        down_2 = F.upsample(down_2, (32,32))
        concat_feat = torch.cat([x, down_16, down_8, down_4, down_2], dim=1)
        
        x = F.leaky_relu(self.conv3(concat_feat), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        z = self.conv7(x)
        return z



if __name__ == '__main__':
    latent_dim = 8
    img_size = 128
    img_shape = (3, img_size, img_size)
    
    generator = Generator(latent_dim, img_shape)
    encoder = Encoder()
    decoder = Decoder()
    discriminator = MultiDiscriminator()

    x = torch.ones((1,3,128,128))
    outputs = discriminator.forward(x)
    codes, feats = encoder(x)

    pdb.set_trace()










