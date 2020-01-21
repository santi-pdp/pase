#!/bin/bash


python -u -W ignore train.py --batch_size 32 --epoch 200 --save_path ckpt_liteGEnhancement_PASE_QRNN512_emb100 \
       	--num_workers 30 --warmup 10000000 --net_cfg cfg/workers_bigcombo.cfg \
	--fe_cfg cfg/PASE_dense_QRNN512_emb100.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root /veu4/usuaris26/spascual/DB/LibriSpeech_50h/wav_sel/ \
	--stats data/librispeech_bigcombo_stats.pkl --lr_mode poly \
	--chunk_size 16000 --tensorboard True --log_freq 50 \
	--dtrans_cfg data/GEnhancement/distortions_SEGANnoises.cfg --seed 91



#python -u -W ignore train.py --batch_size 16 --epoch 15 --save_path ckpt_GEnhancement_PASE_QRNN-512 \
#       	--num_workers 16 --warmup 10000000 --net_cfg cfg/workers5reg_10L2regularizer_allder2_nooverlap.cfg \
#	--fe_cfg cfg/PASE_dense_QRNN512.cfg --do_eval --data_cfg data/GEnhancement/librispeech_contaminated.cfg \
#	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/GEnhancement/LibriSpeech/ \
#	--stats data/librispeech_clean_allder2.pkl --lr_mode poly \
#	--chunk_size 16000 --tensorboard True --dataset GenhancementDataset --log_freq 50


#python -u -W ignore train.py --batch_size 32 --epoch 30 --save_path ckpt_GEnhancement_PASE_QRNN \
#       	--num_workers 10 --warmup 10000000 --net_cfg cfg/workers5reg_10L2regularizer_allder2_nooverlap.cfg \
#	--fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval --data_cfg data/GEnhancement/librispeech_contaminated.cfg \
#	--min_lr 0.0005 --fe_lr 0.0005 --data_root data/GEnhancement/LibriSpeech/ \
#	--stats data/librispeech_clean_allder2.pkl --lr_mode poly \
#	--chunk_size 16000 --tensorboard True --dataset GenhancementDataset --log_freq 50
