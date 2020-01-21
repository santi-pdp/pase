#!/bin/bash


#kernprof -v -l train.py --batch_size 32 --epoch 50 --save_path ckpt_PASE_50h_newMI_noW_denseskips_QRNN_100addrev \

#export SLURM_TMPDIR=/disks/data1/pawel/repos/jsalt_pase_data
export CUDA_VISIBLE_DEVICES=1

python -u train.py --batch_size 20 --epoch 40 \
        --save_path ckpt_PASE_libri_swbd_newMI_noW_CATdenseskips_QRNN_addrev_tmp \
        --num_workers 4 --warmup 10000000 --net_cfg cfg/workers.cfg \
        --fe_cfg cfg/PASE_concatdense_QRNN.cfg --do_eval \
        --min_lr 0.0005 --fe_lr 0.0005  \
        --stats data/librispeech_swbd_100h_stats.pkl --lrdec_step 15 --lrdecay 0.5 \
        --chunk_size 32000 --random_scale True --log_grad_keys rnn \
        --tensorboard True \
        --cchunk_prior \
        --data_root data/LibriSpeech_50h/wav_sel \
        --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
        --dtrans_cfg cfg/distortions/all_pawel.cfg \
        --dataset LibriSpeechSegTupleWavDataset \
        --data_root . \
        --data_cfg data/swbd1_30k.cfg \
        --dtrans_cfg cfg/distortions/all_pawel.cfg \
        --dataset LibriSpeechSegTupleWavDataset \
        --zero_speech_p 0 --zero_speech_p 0
