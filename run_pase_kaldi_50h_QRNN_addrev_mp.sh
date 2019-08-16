#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_QRNN_100addrev_dev-version \
python -u train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_libri_kaldimfcc_plp \
       	--num_workers 16 --warmup 10000000 --net_cfg cfg/workers_kaldi_plp.cfg \
	--fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval \
        --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
	--min_lr 0.0005 --fe_lr 0.0005 \
        --data_root /tmp-corpora/LibriSpeech_50h/wav_sel \
	--dtrans_cfg cfg/distortions/all_pawel.cfg --seed 100 \
	--stats data/librispeech_50h_stats_kaldimfcc_kaldiplp.pkl --lr_mode poly \
	--chunk_size 32000 --random_scale 1 --tensorboard True \
        --kaldi_root /disks/data1/pawel/repos/kaldi

# --log_grad_keys rnn \

