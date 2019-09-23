#!/bin/bash
#SBATCH -J TMAT_hyper
#SBATCH -o log/TMAT_hyper
#SBATCH --mem=64GB
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH --gres=gpu -C V100

nvidia-smi

#module load cuda anaconda

echo "copying data from /scratch":

#cp -r /scratch/jzhong9/data/LibriSpeech_50h /tmp

echo "done!"

source activate myenv

conda info -e

python -u  train.py --batch_size 32 --epoch 200 --save_path /scratch/jzhong9/models/TMAT_V100_best_MAT_deeper \
       --num_workers 12 --warmup 10000000 --net_cfg cfg/workers_best_matconv.cfg \
       --fe_cfg cfg/PASE_TMAT.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
       --min_lr 0.0005 --fe_lr 0.0005 --data_root /scratch/jzhong9/data/LibriSpeech_50h/wav_sel \
       --dtrans_cfg cfg/distortions/half.cfg \
       --stats data/libri_bestder2_jianyuan.pkl \
       --chunk_size 32000 \
       --random_scale True \
       --backprop_mode hyper_volume --delta 1.15 \
       --lr_mode poly \
       --tensorboard True \
#       --fbanks_der_order 2 --gammatone_der_order 2 --LPS_der_order 2 \
#       --mfccs_order 13 --mfccs_der_order 2
#       --sup_exec ./sup_cmd.txt --sup_freq 10 --log_freq 100

