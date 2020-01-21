#!/bin/bash


#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_denseskips_QRNN_100addrev \

python -u train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_QRNN_100addrev_overlapMinion_zerospeech \
       	--num_workers 4 --warmup 10000000 --net_cfg cfg/workers_overlap.cfg \
	--fe_cfg cfg/PASE_concatdense_QRNN.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/all/ \
	--dtrans_cfg cfg/distortions/100addrev.cfg \
	--stats data/librispeech_50h_stats.pkl \
	--chunk_size 32000 --random_scale True --tensorboard True --log_grad_keys rnn \
	--cchunk_prior  --lr_mode poly --zerospeech_cfg cfg/zerospeech.cfg

