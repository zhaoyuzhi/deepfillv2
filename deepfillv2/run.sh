python train.py \
--baseroot '/mnt/lustre/zhaoyuzhi/dataset/ILSVRC2012_train_256' \
--save_path './models' \
--sample_path './samples' \
--gan_type 'WGAN' \
--multi_gpu False \
--cudnn_benchmark True \
--checkpoint_interval 1 \
--multi_gpu True \
--load_name '' \
--epochs 41 \
--batch_size 4 \
--lr_g 1e-4 \
--lr_d 4e-4 \
--lambda_l1 100 \
--lambda_perceptual 10 \
--lambda_gan 1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 8 \
--in_channels 4 \
--out_channels 3 \
--latent_channels 64 \
--pad_type 'zero' \
--activation 'lrelu' \
--norm 'in' \
--init_type 'xavier' \
--init_gain 0.02 \
--imgsize 256 \
--mask_type 'free_form' \
--margin 10 \
--mask_num 20 \
--bbox_shape 30 \
--max_angle 4 \
--max_len 40 \
--max_width 10 \