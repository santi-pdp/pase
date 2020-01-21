#!/bin/bash


python -u train.py --batch_size 16 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_EMB256_2xQRNN_lnorm_100addrev_bsz16 \
       	--num_workers 4 --warmup 10000000 --net_cfg cfg/workers.cfg \
	--fe_cfg cfg/PASE_dense_emb256_lnorm_2xQRNN.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/all/ \
	--dtrans_cfg cfg/distortions/100addrev.cfg \
	--stats data/librispeech_50h_stats.pkl \
	--chunk_size 32000 --random_scale True --tensorboard True --log_grad_keys rnn \
	--cchunk_prior  --lr_mode poly


	#--chunk_size 16000 --random_scale --log_grad_keys rnn \
