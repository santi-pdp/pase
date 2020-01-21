#!/bin/bash


python -u train.py --save_path ckpts_noisyPASE/ckpt_reconPASE_LSGAN \
	--noise_folder data/LibriSpeech/noisy_Librispeech_spkid_sel \
	--whisper_folder data/LibriSpeech/whisper_Librispeech_spkid_sel \
	--distortion_p 0.4 --dtrans_cfg cfg/distortions/all.cfg \
	--net_cfg cfg/all_for_distorted_adversarial.cfg \
	--fe_cfg cfg/PASE_RF6250.cfg \
	--do_eval --stats data/librispeech_stats_nochunks.pkl \
	--trans_cache data/LibriSpeech/aco \
	--num_workers 3 --seed 1010 --batch_size 15 \
	--batch_acum 10 --warmup 100000000 \
	--fe_lr 0.0001 --min_lr 0.0001

#	--pretrained_ckpt ckpts_noisyPASE/ckpt_denoising_pretrain_pase_bsz15-acum10/fullmodel_e235.ckpt_shiftedminions \

