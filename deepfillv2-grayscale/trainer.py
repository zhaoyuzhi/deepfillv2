import time
import datetime
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import itertools

import network
import dataset
import utils

def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path(opt.save_path)
    utils.check_path(opt.sample_path)

    # Build networks
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'GrayInpainting_epoch%d_batchsize%d.pth' % (epoch+20, opt.batch_size)
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, mask) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            mask = mask.cuda()                                              # out: [B, 1, 256, 256]

            # forward propagation
            optimizer_g.zero_grad()
            out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

            # Mask L1 Loss
            MaskL1Loss = L1Loss(out_wholeimg, grayscale)

            # Compute losses
            loss = MaskL1Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1))

def Trainer_GAN(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Build path folder
    utils.check_path(opt.save_path)
    utils.check_path(opt.sample_path)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'GrayInpainting_GAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, mask) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            mask = mask.cuda()                                              # out: [B, 1, 256, 256]

            # LSGAN vectors
            valid = Tensor(np.ones((grayscale.shape[0], 1, 8, 8)))
            fake = Tensor(np.zeros((grayscale.shape[0], 1, 8, 8)))

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_d.zero_grad()

            # forward propagation
            out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(grayscale, mask)
            # Overall Loss and optimize
            loss_fake = MSELoss(fake_scalar, fake)
            loss_true = MSELoss(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()

            # ----------------------------------------
            #             Train Generator
            # ----------------------------------------
            optimizer_g.zero_grad()

            # forward propagation
            out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

            # Mask L1 Loss
            MaskL1Loss = L1Loss(out_wholeimg, grayscale)

            # GAN Loss
            fake_scalar = discriminator(out_wholeimg, mask)
            MaskGAN_Loss = MSELoss(fake_scalar, valid)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            out_3c = torch.cat((out_wholeimg, out_wholeimg, out_wholeimg), 1)
            grayscale_3c = torch.cat((grayscale, grayscale, grayscale), 1)
            out_featuremaps = perceptualnet(out_3c)
            gt_featuremaps = perceptualnet(grayscale_3c)
            PerceptualLoss = L1Loss(out_featuremaps, gt_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * MaskL1Loss + opt.lambda_perceptual * PerceptualLoss + opt.lambda_gan * MaskGAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [Perceptual Loss: %.5f] [D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), PerceptualLoss.item(), loss_D.item(), MaskGAN_Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_d, (epoch + 1), opt, opt.lr_d)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1))
