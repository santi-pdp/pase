#!/bin/bash


python -u train.py --batch_size 10 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_LSTM512_all_overlap_gap \
       	--num_workers 4 --warmup 10000000 --net_cfg cfg/workers_overlap_gap.cfg \
	--fe_cfg cfg/PASE_dense_BLSTM.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/all/ \
	--dtrans_cfg cfg/distortions/all_x26.cfg --seed 100 \
	--stats data/librispeech_50h_stats.pkl --lr_mode poly \
	--chunk_size 32000 --random_scale True --tensorboard True 

# --log_grad_keys rnn \

