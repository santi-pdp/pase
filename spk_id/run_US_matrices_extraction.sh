#!/bin/bash


# DATASET params
export SPK2IDX="../data/interface/inter1en/interface_dict.npy"
export DATA_ROOT="../data/interface/inter1en/all_wav"
export TRAIN_GUIA="../data/interface/inter1en/interface_tr.scp"
export TEST_GUIA="../data/interface/inter1en/interface_te.scp"
# root to store all supervised ckpts
export SAVE_ROOT="UE_interface_ckpts/"

#bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_multitask-1x256_lrs00005/ UE_MULTITASK
#bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_lps-mfcc-MI-1x256_lrs00005_fastSincConv/ UE_LPSMFCCMI
bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_lps-mfcc-MI-CMI-1x256_lrs00005/ UE_LPSMFCCMICMI
#TODO: Must finish experiments of multitask+CMI first


# DATASET params
export SPK2IDX="../data/VCTK/spk_id/vctk_dict.npy"
export DATA_ROOT="../data/VCTK/spk_id/all_trimmed_wav16"
export TRAIN_GUIA="../data/VCTK/spk_id/vctk_tr.scp"
export TRAIN_GUIA="../data/VCTK/spk_id/vctk_te.scp"
# root to store all supervised ckpts
export SAVE_ROOT="UE_vctk_ckpts/"

bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_lps-mfcc-MI-1x256_lrs00005_fastSincConv/ UE_VCTK_LPSMFCCMI
bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_multitask-1x256_lrs00005/ UE_VCTK_MULTITASK
bash build_us_matrix.sh ../ckpts_unsup-librispeech/smallFE-e100_lps-mfcc-MI-CMI-1x256_lrs00005/ UE_VCTK_LPSMFCCMICMI
#TODO: Must finish experiments of multitask+CMI first
