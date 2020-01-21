#!/bin/bash

python -u  train.py --batch_size 10 --epoch 50 --save_path ckpts_JSALT2019/ASPP \
              --num_workers 8 --warmup 10000000 --net_cfg cfg/workers_aspp.cfg \
              --fe_cfg cfg/PASE_aspp_res1d.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
              --min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/wav_sel/ \
              --dtrans_cfg cfg/distortions/half_x26.cfg \
              --stats data/librispeech_50h_stats.pkl \
              --chunk_size 32000 \
              --tensorboard True \
              --random_scale True\
              --lr_mode poly
