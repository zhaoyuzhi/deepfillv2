# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def forward(size, root, model):
    # pre-processing, let all the images are in RGB color space
    img = Image.open(root)
    img = img.resize((size, size), Image.ANTIALIAS).convert('RGB')
    img = np.array(img).astype(np.float64)
    # define a mask
    mask = np.zeros([size, size, 1], dtype = np.float64)
    if size == 144:
        center = np.ones([100, 100, 1], dtype = np.float64)
        mask[22:122, 22:122, :] = center
    elif size == 200:
        center = np.ones([144, 144, 1], dtype = np.float64)
        mask[28:172, 28:172, :] = center
    elif size == 256:
        center = np.ones([200, 200, 1], dtype = np.float64)
        mask[28:228, 28:228, :] = center
    maskimg = (img * mask) / 255
    maskimg = maskimg.astype(np.float32)
    maskimg = transforms.ToTensor()(maskimg)
    maskimg = maskimg.reshape([1, 3, size, size])
    mask = mask.astype(np.float32)
    mask = transforms.ToTensor()(mask)
    mask = mask.reshape([1, 1, size, size])
    maskimg = torch.cat((maskimg, mask), 1).cuda()
    # get the output
    output = model(maskimg)
    # transfer to image
    output = output.cpu().detach().numpy().reshape([3, size, size])
    output = output.transpose(1, 2, 0)
    output = output * 255
    output = np.array(output, dtype = np.uint8)
    return output

if __name__ == "__main__":

    size = 256
    root = 'C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\COCO2014_val_256\\COCO_val2014_000000000285.jpg'
    #model = torch.load('Pre_PRPGAN_1st_epoch5_batchsize8.pth')
    model = torch.load('TestNet_epoch10_batchsize8.pth')

    output = forward(size, root, model)
    img = Image.fromarray(output)
    img.show()
