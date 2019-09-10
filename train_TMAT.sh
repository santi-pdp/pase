#!/bin/bash
#SBATCH -J aspp
#SBATCH -o log/aspp_32000
#SBATCH --mem=32GB
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu -C K80

nvidia-smi

python -u  train.py --batch_size 24 --epoch 200 --save_path ~/jsalt/models/TMAT_modified \
       --num_workers 6 --warmup 10000000 --net_cfg cfg/workers5reg_r3.cfg \
       --fe_cfg cfg/PASE_TMAT.cfg --do_eval --data_cfg data/LibriSpeech_50h/LibriSpeech_50h/librispeech_data_50h.cfg \
       --min_lr 0.0005 --fe_lr 0.0005 --data_root data/LibriSpeech_50h/LibriSpeech_50h/wav_sel \
       --dtrans_cfg cfg/distortions/half.cfg \
       --stats data/librispeech_50h_stats_libri_allder2.pkl \
       --chunk_size 32000 \
       --random_scale True \
       --backprop_mode hyper_volume --delta 1.15\
       --lr_mode poly \
       --tensorboard True \
#       --fbanks_der_order 2 --gammatone_der_order 2 --LPS_der_order 2 \
#       --mfccs_order 13 --mfccs_der_order 2
#       --sup_exec ./sup_cmd.txt --sup_freq 10 --log_freq 100

