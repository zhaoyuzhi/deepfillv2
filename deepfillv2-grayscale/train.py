import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'the type of GAN for training')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--multi_gpu', type = bool, default = True, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0, 1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 100, help = 'interval between model checkpoints')
    parser.add_argument('--finetune_path', type = str, default = "", help = 'the load name of models')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 101, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 2e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 2e-4, help = 'Adam: learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 25, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 10, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 1, help = 'the parameter of perceptual loss')
    parser.add_argument('--lambda_gan', type = float, default = 0.1, help = 'the parameter of GAN loss')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of cpu threads to use during batch generation')
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
    parser.add_argument('--baseroot', type = str, default = "C:\\Users\\yzzha\\Desktop\\dataset\\1", \
        help = 'the base training folder for inpainting network')
    parser.add_argument('--imgsize', type = int, default = 256, help = 'size of image')
    # mask parameters
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 20, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 40, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 2, help = 'parameter of width for free form mask')
    opt = parser.parse_args()
    print(opt)
    
    '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
    
    # Enter main function
    import trainer
    if opt.pre_train == True:
        trainer.Trainer(opt)
    else:
        trainer.Trainer_GAN(opt)
    