# deepfillv2

The PyTorch implementations and guideline for Gated Convolution based on ICCV 2019 oral paper: free-form inpainting (deepfillv2).

## 1 Implementations

Before running it, please ensure the environment is `Python 3.6` and `PyTorch 1.0.1`.

### 1.1  Train

If you train it from scratch, please specify following hyper-parameters. For other parameters, we recommend the default settings.

```bash
python train.py     --epochs 40
                    --lr_g 0.0001
                    --batch_size 4
                    --perceptual_param 10
                    --gam_param 0.01
                    --baseroot [the path of TrainingPhoneRaw]
                    --mask_type 'free_form' [or 'single_bbox' or 'bbox']
                    --imgsize 256
```
```bash
if you have more than one GPU, please change following codes:
python train.py     --multi_gpu True
                    --gpu_ids [the ids of your multi-GPUs]
```

### 1.2  Test

At testing phase, please download the pre-trained [model]() first.

For small image patches:
```bash
python test.py 	    --load_name '*.pth' (please ensure the pre-trained model is in same path)
                    --baseroot [the path of TestingPhoneRaw]
                    --mask_type 'free_form' [or 'single_bbox' or 'bbox']
                    --imgsize 256
```

There are some examples:

<img src="./images/result1.png" width="1000"/>

<img src="./images/result2.png" width="1000"/>

The corresponding ground truth is:

<img src="./images/gt.png" width="1000"/>
