import argparse
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'the type of GAN for training')
    parser.add_argument('--finetune_path', type = str, \
        default = "./models/GrayInpainting_epoch20_batchsize16.pth", \
            help = 'the load name of models')
    parser.add_argument('--savepath', type = str, default = "./results", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'test batch size')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'num of workers')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'input mask')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256", \
            help = 'the testing folder')
    parser.add_argument('--maskroot', type = str, \
        default = "C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256_mask", \
            help = 'the mask folder')
    parser.add_argument('--saveroot', type = str, \
        default = "./results", \
            help = 'the saving folder')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.ValidationSet_with_Known_Mask(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.savepath)

    # forward
    for i, (grayscale, mask, imgname) in enumerate(test_loader):
        '''
        # Show
        masked_img = grayscale * (1 - mask) + mask
        masked_img = masked_img[0, 0, :, :].numpy() * 255.0
        masked_img = masked_img.astype(np.uint8)
        cv2.imshow('show', masked_img)
        cv2.waitKey(0)
        '''

        # To device
        grayscale = grayscale.cuda()                                        # out: [B, 1, 256, 256]
        mask = mask.cuda()                                                  # out: [B, 1, 256, 256]
        print(i, imgname[0])

        # Forward propagation
        with torch.no_grad():
            fake_target = generator(grayscale, mask)                        # out: [B, 1, 256, 256]
        fake_target = grayscale * (1 - mask) + fake_target * mask           # in range [0, 1]

        # Save
        fake_target = fake_target.clone().data[0, 0, :, :].cpu().numpy()
        fake_target = (fake_target * 255.0).astype(np.uint8)
        fake_target = cv2.cvtColor(fake_target, cv2.COLOR_BGR2RGB)
        save_img_path = os.path.join(opt.savepath, imgname[0])
        print(save_img_path)
        cv2.imwrite(save_img_path, fake_target)
        '''
        cv2.imshow('show', fake_target)
        cv2.waitKey(0)
        '''
        