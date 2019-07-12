#!/bin/bash
#SBATCH -J hyper1.1_K80
#SBATCH -o log/hyper_volume_1.1.K80.out
#SBATCH --mem=32GB
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -C K80

python -u  train.py --batch_size 32 --epoch 50 --save_path /export/team-mic/zhong/hyper_noise \
       --num_workers 8 --warmup 10000000 --net_cfg cfg/workers.cfg \
       --fe_cfg cfg/PASE_dense.cfg --do_eval --data_cfg /export/corpora/LibriSpeech_50h/librispeech_data_50h.cfg \
       --min_lr 0.0005 --fe_lr 0.0005 --data_root /export/corpora/LibriSpeech_50h/wav_sel \
       --dtrans_cfg cfg/distortions/all.cfg \
       --stats data/librispeech_50h_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
       --chunk_size 32000 \
       --random_scale \
       --backprop_mode hyper_volume --delta 1.1 \
       --tensorboard True \
       --sup_exec sup_cmd_mila.txt --sup_freq 10
