import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = "/home/alien/Documents/LINTingyu/inpainting/validation", help = 'the testing folder')
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 10, help = 'parameter of width for free form mask')
    # Other parameters
    parser.add_argument('--batch_size', type = int, default = 1, help = 'test batch size, always 1')
    parser.add_argument('--load_name', type = str, default = 'deepfillNet_epoch4_batchsize4.pth', help = 'test model name')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #       Initialize testing dataset
    # ----------------------------------------

    # Define the dataset
    testset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(testset))

    # Define the dataloader
    dataloader = DataLoader(testset, batch_size = opt.batch_size, pin_memory = True)

    # ----------------------------------------
    #                 Testing
    # ----------------------------------------

    model = torch.load(opt.load_name)

    for batch_idx, (img, mask) in enumerate(dataloader):

        # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        masked_img = img * (1 - mask)
        fake1, fake2 = model(masked_img, mask)

        # forward propagation
        fusion_fake1 = img * (1 - mask) + fake1 * mask                      # in range [-1, 1]
        fusion_fake2 = img * (1 - mask) + fake2 * mask                      # in range [-1, 1]

        # convert to visible image format
        img = img.cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        img = (img + 1) * 128
        img = img.astype(np.uint8)
        fusion_fake1 = fusion_fake1.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        fusion_fake1 = (fusion_fake1 + 1) * 128
        fusion_fake1 = fusion_fake1.astype(np.uint8)
        fusion_fake2 = fusion_fake2.detach().cpu().numpy().reshape(3, opt.imgsize, opt.imgsize).transpose(1, 2, 0)
        fusion_fake2 = (fusion_fake2 + 1) * 128
        fusion_fake2 = fusion_fake2.astype(np.uint8)

        # show
        show_img = np.concatenate((img, fusion_fake1, fusion_fake2), axis = 1)
        r, g, b = cv2.split(show_img)
        show_img = cv2.merge([b, g, r])
        cv2.imshow('comparison.jpg', show_img)
        cv2.imwrite('result_%d.jpg' % batch_idx, show_img)
