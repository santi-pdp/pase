#!/bin/bash


python -u -W ignore train.py --batch_size 16 --epoch 156 --save_path /veu4/santi.pasqual/ckpt_miqvoxceleb_QRNN1024-emb100 \
       	--num_workers 10 --warmup 10000000 --net_cfg cfg/workers5reg_10L2regularizer_allder2_nooverlap.cfg \
	--fe_cfg cfg/PASE_dense_QRNN1024_emb100.cfg --do_eval --data_cfg data/miqvoxceleb_data.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 --data_root /veu/miquel.india/databases/ \
	--dtrans_cfg cfg/distortions/100addrev_25overlap.cfg \
	--stats data/miqvoxceleb_allder2.pkl --lr_mode poly \
	--chunk_size 16000 --tensorboard True --log_freq 200


