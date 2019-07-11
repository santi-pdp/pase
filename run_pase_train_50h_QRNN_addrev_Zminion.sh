#!/bin/bash


#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_denseskips_QRNN_100addrev \

python -u train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_denseskips_QRNN_100addrev_Zmatching_chk24000 \
       	--num_workers 4 --warmup 0 --net_cfg cfg/workers_Zmatching.cfg \
	--fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--z_lr 0.001 --min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/all/ \
	--dtrans_cfg cfg/distortions/100addrev.cfg \
	--stats data/librispeech_50h_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
	--chunk_size 24000 --random_scale --log_grad_keys rnn \
	--cchunk_prior 

