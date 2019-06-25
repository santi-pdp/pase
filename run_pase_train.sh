#!/bin/bash

python -u train.py --batch_size 32 --epoch 150 --save_path ckpt_PASE \
       	--num_workers 0 --warmup 10000000 --net_cfg cfg/workers.cfg \
	--fe_cfg cfg/PASE.cfg --do_eval --data_cfg data/librispeech_data.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech/Librispeech_spkid_sel/ \
	--stats data/librispeech_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
	--trans_cache data/LibriSpeech/aco --preload_wav
