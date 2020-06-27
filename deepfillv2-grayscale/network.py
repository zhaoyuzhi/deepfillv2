import torch
import torch.nn as nn
import torch.nn.init as init

from network_module import *

def weights_init(net, init_type = 'kaiming', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GrayInpaintingNet(nn.Module):
    def __init__(self, opt):
        super(GrayInpaintingNet, self).__init__()
        # Downsampling
        self.down1 = GatedConv2d(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = 'none')
        self.down2 = GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down3 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.down4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Bottleneck
        self.b1 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b2 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b3 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b4 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b5 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b6 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b7 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.b8 = GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        # Upsampling
        self.up1 = TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up2 = GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up3 = TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm_g)
        self.up4 = GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad, activation = 'sigmoid', norm = 'none')
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        masked_img = img * (1 - mask) + mask                            # in: batch * 1 * 256 * 256
        fusion = torch.cat((masked_img, mask), 1)                       # in: batch * 2 * 256 * 256
        # network forward part
        out = self.down1(fusion)                                        # out: batch * 64 * 256 * 256
        out = self.down2(out)                                           # out: batch * 128 * 128 * 128
        out = self.down3(out)                                           # out: batch * 256 * 128 * 128
        out = self.down4(out)                                           # out: batch * 256 * 64 * 64
        out = self.b1(out)                                              # out: batch * 256 * 64 * 64
        out = self.b2(out)                                              # out: batch * 256 * 64 * 64
        out = self.b3(out)                                              # out: batch * 256 * 64 * 64
        out = self.b4(out)                                              # out: batch * 256 * 64 * 64
        out = self.b5(out)                                              # out: batch * 256 * 64 * 64
        out = self.b6(out)                                              # out: batch * 256 * 64 * 64
        out = self.b7(out)                                              # out: batch * 256 * 64 * 64
        out = self.b8(out)                                              # out: batch * 256 * 64 * 64
        out = self.up1(out)                                             # out: batch * 128 * 128 * 128
        out = self.up2(out)                                             # out: batch * 128 * 128 * 128
        out = self.up3(out)                                             # out: batch * 64 * 256 * 256
        out = self.up4(out)                                             # out: batch * 3 * 256 * 256
        # final output
        return out

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 8 * 8
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.mask_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm_d, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                              # out: batch * 64 * 256 * 256
        x = self.block2(x)                                              # out: batch * 128 * 128 * 128
        x = self.block3(x)                                              # out: batch * 256 * 64 * 64
        x = self.block4(x)                                              # out: batch * 256 * 32 * 32
        x = self.block5(x)                                              # out: batch * 256 * 16 * 16
        x = self.block6(x)                                              # out: batch * 1 * 8 * 8
        return x

# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return x
