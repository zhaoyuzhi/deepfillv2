import argparse
import os
import torch
import numpy as np
import cv2

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
        default = "./models/GrayInpainting_epoch20_batchsize8.pth", \
            help = 'the load name of models')
    parser.add_argument('--val_path', type = str, default = "./validation", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
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
            help = 'the base training folder for inpainting network')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    # mask parameters
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    opt = parser.parse_args()

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.InpaintDataset_val(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.val_path)

    # forward
    for i, (grayscale, mask, imgname) in enumerate(test_loader):

        # To device
        grayscale = grayscale.cuda()                                        # out: [B, 1, 256, 256]
        mask = mask.cuda()                                                  # out: [B, 1, 256, 256]
        print(i, imgname[0])

        # Forward propagation
        with torch.no_grad():
            fake_target = generator(grayscale, mask)                        # out: [B, 1, 256, 256]

        # Save
        fake_target = fake_target.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        fake_target = (fake_target * 255.0).astype(np.uint8)
        fake_target = cv2.cvtColor(fake_target, cv2.COLOR_BGR2RGB)
        save_img_path = os.path.join(opt.val_path, imgname[0])
        cv2.imwrite(imgname[0], fake_target)
        