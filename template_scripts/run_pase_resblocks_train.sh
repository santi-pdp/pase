#!/bin/bash

python -u train.py --batch_size 32 --epoch 150 --save_path ckpt_PASE_resnet_nodil \
       	--num_workers 4 --warmup 10000000 --net_cfg cfg/workers.cfg \
	--fe_cfg cfg/PASE_resnet_nodil.cfg --do_eval --data_cfg data/librispeech_data.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech/Librispeech_spkid_sel/ \
	--stats data/librispeech_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
	--preload_wav
