#!/bin/bash


#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_denseskips_QRNN_100addrev \

#export SLURM_TMPDIR=/disks/data1/pawel/repos/jsalt_pase_data
#export CUDA_VISIBLE_DEVICES=1

python -u train.py --batch_size 32 --epoch 40 \
        --save_path ckpt_PASE_libri_revno_ami_ihm_revno \
        --num_workers 4 --warmup 10000000 --net_cfg cfg/workers.cfg \
        --fe_cfg cfg/PASE_concatdense_QRNN.cfg --do_eval \
        --min_lr 0.0005 --fe_lr 0.0005  \
        --stats data/libri_ami_ihm_stats.pkl --lrdec_step 20 --lrdecay 0.5 \
        --chunk_size 32000 --random_scale True --log_grad_keys rnn \
        --tensorboard True \
        --cchunk_prior \
        --data_root data/LibriSpeech_50h/wav_sel \
        --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
        --dtrans_cfg cfg/distortions/all.cfg \
        --dataset LibriSpeechSegTupleWavDataset \
        --data_root /export/team-mic/corpora/ami \
        --data_cfg data/ami_data_ihm_sdm1357.cfg \
        --dtrans_cfg cfg/distortions/all.cfg \
        --dataset AmiSegTupleWavDataset

