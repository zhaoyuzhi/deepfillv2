#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Takes exactly 1 argument"
    echo "   sh ./start_script.sh /path/to/training/data"
fi

BASEROOT=$1
echo Training with $BASEROOT 

python train.py \
	--epochs 40 \
	--lr_g 0.0001 \
	--batch_size 4 \
	--perceptual_param 10 \
	--gan_param 0.01 \
	--baseroot $BASEROOT \
	--mask_type 'free_form' \
	--imgsize 256