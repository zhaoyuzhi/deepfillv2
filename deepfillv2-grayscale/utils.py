import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
import dataset

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GrayInpaintingNet(opt)
    print('Generator is created!')
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % opt.finetune_path)
    else:
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Pre-trained VGG-16
    vgg16 = torch.load('vgg16_pretrained.pth')
    # Get the first 16 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Update the parameters
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def sample(grayscale, mask, out, save_folder, epoch):
    # to cpu
    grayscale = grayscale[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                     # 256 * 256 * 1
    mask = mask[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                               # 256 * 256 * 1
    out = out[0, :, :, :].data.cpu().numpy().transpose(1, 2, 0)                                 # 256 * 256 * 1
    # process
    masked_img = grayscale * (1 - mask) + mask                                                  # 256 * 256 * 1
    masked_img = np.concatenate((masked_img, masked_img, masked_img), axis = 2)                 # 256 * 256 * 3 (√)
    masked_img = (masked_img * 255).astype(np.uint8)
    grayscale = np.concatenate((grayscale, grayscale, grayscale), axis = 2)                     # 256 * 256 * 3 (√)
    grayscale = (grayscale * 255).astype(np.uint8)
    mask = np.concatenate((mask, mask, mask), axis = 2)                                         # 256 * 256 * 3 (√)
    mask = (mask * 255).astype(np.uint8)
    out = np.concatenate((out, out, out), axis = 2)                                             # 256 * 256 * 3 (√)
    out = (out * 255).astype(np.uint8)
    # save
    img = np.concatenate((grayscale, mask, masked_img, out), axis = 1)
    imgname = os.path.join(save_folder, str(epoch) + '.png')
    cv2.imwrite(imgname, img)
    
def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim
