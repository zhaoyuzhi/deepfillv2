import os
import math
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

import utils

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class DomainTransferDataset(Dataset):
    def __init__(self, opt):
        super(DomainTransferDataset, self).__init__()
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A)
        self.imglist_B = utils.get_files(opt.baseroot_B)
        self.len_A = len(self.imglist_A)
        self.len_B = len(self.imglist_B)
    
    def imgcrop(self, img):
        H, W = img.shape
        # scaled size should be greater than opts.crop_size
        if H < W:
            if H < self.opt.crop_size:
                H_out = self.opt.crop_size
                W_out = int(math.floor(W * float(H_out) / float(H)))
                img = cv2.resize(img, (W_out, H_out))
        else: # W_out < H_out
            if W < self.opt.crop_size:
                W_out = self.opt.crop_size
                H_out = int(math.floor(H * float(W_out) / float(W)))
                img = cv2.resize(img, (W_out, H_out))
        # randomly crop
        rand_h = random.randint(0, max(0, H - self.opt.imgsize))
        rand_w = random.randint(0, max(0, W - self.opt.imgsize))
        img = img[rand_h:rand_h + self.opt.imgsize, rand_w:rand_w + self.opt.imgsize, :]
        return img

    def __getitem__(self, index):

        ## Image A
        random_A = random.randint(0, self.len_A - 1)
        imgpath_A = self.imglist_A[random_A]
        img_A = cv2.imread(imgpath_A, cv2.IMREAD_GRAY)
        # image cropping
        img_A = self.imgcrop(img_A)
        
        ## Image B
        random_B = random.randint(0, self.len_B - 1)
        imgpath_B = self.imglist_B[random_B]
        img_B = cv2.imread(imgpath_B, cv2.IMREAD_GRAY)
        # image cropping
        img_B = self.imgcrop(img_B)

        # To tensor (grayscale)
        img_A = torch.from_numpy(img_A.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        img_B = torch.from_numpy(img_B.astype(np.float32) / 255.0).unsqueeze(0).contiguous()

        return img_A, img_B
    
    def __len__(self):
        return min(self.len_A, self.len_B)

class InpaintDataset(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # mask
        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
        mask = torch.from_numpy(mask).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        #img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # grayscale: 1 * 256 * 256; mask: 1 * 256 * 256
        return grayscale, mask

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

class InpaintDataset_val(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.imglist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        # image read
        imgname = self.imglist[index]                                       # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('RGB')                            # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)
        # grayish
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # mask
        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
        mask = torch.from_numpy(mask).contiguous()

        # normalization
        grayscale = torch.from_numpy(grayscale.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        #img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # grayscale: 1 * 256 * 256; mask: 1 * 256 * 256
        return grayscale, mask, imgname

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

class ValidationSet_with_Known_Mask(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.namelist = utils.get_jpgs(opt.baseroot)

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # image
        imgname = self.namelist[index]                                      # name of one image
        imgpath = os.path.join(self.opt.baseroot, imgname)                  # path of one image
        img = Image.open(imgpath).convert('L')                              # read one image (RGB)
        img = np.array(img)                                                 # read one image
        # image resize
        img = cv2.resize(img, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)

        # mask
        maskpath = os.path.join(self.opt.maskroot, imgname)
        mask = Image.open(maskpath).convert('L')
        mask = np.array(mask)                                                 # read one image
        # image resize
        mask = cv2.resize(mask, (self.opt.imgsize, self.opt.imgsize), interpolation = cv2.INTER_CUBIC)

        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return img, mask, imgname
