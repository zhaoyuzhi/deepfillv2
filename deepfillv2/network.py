import torch
import torch.nn as nn
from network_module import *

#-----------------------------------------------
#                   Generator
#-----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, opt):
        super(GatedGenerator, self).__init__()
        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad_type, activation = 'tanh', norm = 'none')
        )
        self.refinement = nn.Sequential(
            # encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # Bottleneck
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            # decoder
            TransposeGatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            TransposeGatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels, opt.out_channels, 7, 1, 3, pad_type = opt.pad_type, activation = 'tanh', norm = 'none')
        )
        
    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # 1 - mask: unmask
        # img * (1 - mask): ground truth unmask region
        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), 1)       # in: [B, 4, H, W]
        first_out = self.coarse(first_in)                       # out: [B, 3, H, W]
        # Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat((second_masked_img, mask), 1)     # in: [B, 4, H, W]
        second_out = self.refinement(second_in)                 # out: [B, 3, H, W]
        return first_out, second_out

#-----------------------------------------------
#                  Discriminator
#-----------------------------------------------
# Input: generated image / ground truth and mask
# Output: patch based region, we set 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels, opt.latent_channels, 7, 1, 3, pad_type = opt.pad_type, activation = opt.activation, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.latent_channels, opt.latent_channels * 2, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.latent_channels * 2, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block4 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block5 = Conv2dLayer(opt.latent_channels * 4, opt.latent_channels * 4, 4, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm, sn = True)
        self.block6 = Conv2dLayer(opt.latent_channels * 4, 1, 4, 2, 1, pad_type = opt.pad_type, activation = 'none', norm = 'none', sn = True)
        
    def forward(self, img, mask):
        # the input x should contain 4 channels because it is a combination of recon image and mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)                                      # out: [B, 64, 256, 256]
        x = self.block2(x)                                      # out: [B, 128, 128, 128]
        x = self.block3(x)                                      # out: [B, 256, 64, 64]
        x = self.block4(x)                                      # out: [B, 256, 32, 32]
        x = self.block5(x)                                      # out: [B, 256, 16, 16]
        x = self.block6(x)                                      # out: [B, 256, 8, 8]
        return x
