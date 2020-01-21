#!/bin/bash

python -u train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_auxsup \
       	--num_workers 1 --warmup 10000000 --net_cfg cfg/workers_weighted.cfg \
	--fe_cfg cfg/PASE.cfg --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root /export/corpora/LibriSpeech_50h/wav_sel \
	--stats data/librispeech_50h_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
	--chunk_size 32000 --sup_exec sup_cmd.txt --sup_freq 1
