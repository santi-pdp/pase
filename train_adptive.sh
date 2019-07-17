#!/bin/bash
#SBATCH -J hyper1.6_v100
#SBATCH -o log/hyper_volume_1.6.V100.out
#SBATCH --mem=32GB
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 8 -C E52695v4
#SBATCH -p gpu
#SBATCH --gres=gpu -C V100


CUDA_VISIBLE_DEVICES=3 python -u train.py --batch_size 32 --epoch 150 --num_workers 8 \
        --save_path /scratch/jzhong9/model_ckpt/adaptive_1_0.1  \
        --net_cfg cfg/workers.cfg \
        --fe_cfg cfg/PASE_dense.cfg \
        --do_eval --data_cfg /export/corpora/LibriSpeech_50h/librispeech_data_50h.cfg --min_lr 0.0005 --fe_lr 0.0005 \
        --data_root /export/corpora/LibriSpeech_50h/wav_sel \
        --stats /export/fs01/zhong/Pase/data/librispeech_50h_stats.pkl \
        --log_freq 100 \
	    --backprop_mode adaptive --temp 1 --alpha 0.1 \
	    --tensorboard True\
        --chunk_size 32000
