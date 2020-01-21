#!/bin/bash
#SBATCH -J hyper1.1_K80
#SBATCH -o log/hyper_volume_1.1.K80.out
#SBATCH --mem=32GB
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -C K80

python -u  train.py --batch_size 5 --epoch 50 --save_path /export/team-mic/zhong/test/print_since_pase \
       --num_workers 8 --warmup 10000000 --net_cfg cfg/workers_overlap.cfg \
       --fe_cfg cfg/PASE_dense_QRNN.cfg --do_eval --data_cfg /export/corpora/LibriSpeech_50h/librispeech_data_50h.cfg \
       --min_lr 0.0005 --fe_lr 0.0005 --data_root /export/corpora/LibriSpeech_50h/wav_sel \
       --dtrans_cfg cfg/distortions/overlap.cfg \
       --stats data/librispeech_50h_stats.pkl --lrdec_step 30 --lrdecay 0.5 \
       --chunk_size 16000 \
       --random_scale True \
       --backprop_mode adaptive --temp 1 --alpha 0.1 \
       --lr_mode step \
       --tensorboard True \
       --att_cfg cfg/attention.cfg --attention_K 40 \
       --sup_exec ./sup_cmd.txt --sup_freq 1
