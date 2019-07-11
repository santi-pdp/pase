#!/bin/bash
#SBATCH -J attention
#SBATCH -o log/att_dense
#SBATCH --mem=32GB
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 8 -C E52695v4
#SBATCH -p gpu
#SBATCH --gres=gpu:1 -C K80

module load cuda
module load anaconda
source activate humor
conda info -e
nvidia-smi

python -u train.py --batch_size 32 --epoch 150 --num_workers 8 \
        --save_path /scratch/jzhong9/model_ckpt/att_dense  \
        --net_cfg cfg/workers.cfg \
        --fe_cfg cfg/PASE_dense.cfg \
        --do_eval --data_cfg /scratch/jzhong9/data/LibriSpeech_50h/librispeech_data_50h.cfg --min_lr 0.0005 --fe_lr 0.0005 \
        --data_root /scratch/jzhong9/data/LibriSpeech_50h/wav_sel \
        --stats /scratch/jzhong9/data/LibriSpeech_50h/librispeech_50h_stats.pkl \
        --log_freq 100 \
	    --att_cfg cfg/attention.cfg --backprop_mode base --tensorboard False