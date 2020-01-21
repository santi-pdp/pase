#!/bin/bash


python -u -W ignore train.py --batch_size 32 --epoch 25 --save_path ckpt_PASE_LibriTTS_5reg_alldis_r3 \
       	--num_workers 25 --warmup 10000000 --net_cfg cfg/workers5reg_r3.cfg \
	--fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval --data_cfg data/libritts_x26.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriTTS_x26/ \
	--dtrans_cfg cfg/distortions/all.cfg \
	--stats data/libritts_stats.pkl --lr_mode poly \
	--chunk_size 24000 --random_scale True --tensorboard True \
	--fe_opt radam --min_opt radam

