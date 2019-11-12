import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class InpaintDataset(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = Image.open(self.imglist[index])
        img = img.resize((self.opt.imgsize, self.opt.imgsize), Image.ANTIALIAS).convert('RGB')
        img = self.transform(img)
        # mask
        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
        # the outputs are entire image and mask, respectively
        return img, mask

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 5 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)