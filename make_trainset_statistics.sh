#!/bin/bash

libri=false
libri_alldeltas=false
genhlibri_allder2_5reg=true
libri_kaldi=false
swbd=false
ami=false
libri_ami=false
libri_ami_sdm=false
libri_ami_swbd=false
chime5=false
chime5_libri=false

$libri && {

python make_trainset_statistics.py \
  --data_root /tmp-corpora/LibriSpeech_50h/wav_sel \
  --data_cfg  /tmp-corpora/LibriSpeech_50h/librispeech_data_50h.cfg \
  --num_workers 5 \
  --kaldi_root /disks/data1/pawel/repos/kaldi \
  --out_file data/librispeech_tmp_stats.pkl

}

$genhlibri_allder2_5reg && {
  der=2
  python make_trainset_statistics.py \
   --data_root data/GEnhancement/LibriSpeech \
   --data_cfg data/GEnhancement/librispeech_clean.cfg \
   --num_workers 10 --max_batches=30 \
   --net_cfg cfg/workers5reg_L2regularizer_allder2.cfg \
   --out_file data/librispeech_clean_allder2.pkl
}

$libri_alldeltas && {
  der=3
  python make_trainset_statistics.py \
   --data_root data/LibriSpeech_50h/wav_sel \
   --data_cfg  data/LibriSpeech_50h/librispeech_data_50h.cfg \
   --num_workers 10 --max_batches=30 \
   --mfccs_order 13 \
   --LPS_der_order $der --gammatone_der_order $der --fbanks_der_order $der --mfccs_der_order $der \
   --mfccs_librosa_order 13 --mfccs_librosa_n_mels 40 --mfccs_librosa_der_order $der \
   --kaldimfccs_num_ceps 13 --kaldimfccs_num_mel_bins 40 --kaldimfccs_der_order $der \
   --kaldi_root /disks/data1/pawel/repos/kaldi \
   --out_file data/librispeech_50h_stats_all_der$der.pkl
}

$libri_kaldi && {
  python make_trainset_statistics.py \
   --data_root /tmp-corpora/LibriSpeech_50h/wav_sel \
   --data_cfg  /tmp-corpora/LibriSpeech_50h/librispeech_data_50h.cfg \
   --num_workers 10 --max_batches=30 \
   --kaldi_root /disks/data1/pawel/repos/kaldi \
   --kaldimfccs_num_mel_bins 40 --kaldimfccs_num_ceps 13 --kaldimfccs_der_order 2 \
   --out_file data/librispeech_50h_stats_kaldimfcc_der.pkl
}

$libri_ami && {

  python make_trainset_statistics.py \
    --num_workers 5 --max_batches 50 \
    --data_root data/LibriSpeech_50h/wav_sel \
    --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
    --dataset PairWavDataset \
    --data_root data/ami \
    --data_cfg data/ami_data_ihm_sdm1357.cfg \
    --dataset AmiSegTupleWavDataset \
    --out_file data/libri_ami_ihm_stats.pkl
}

$libri_ami_sdm && {
   python make_trainset_statistics.py \
    --num_workers 5 --max_batches 50 \
    --data_root data/LibriSpeech_50h/wav_sel \
    --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
    --dataset PairWavDataset \
    --data_root data/ami \
    --data_cfg data/ami_data_ihm_sdm1357.cfg \
    --dataset AmiSegTupleWavDataset \
    --ihm2sdm 1,3,5,7 \
    --out_file data/libri_ami_sdm_stats.pkl
}

$libri_ami_swbd && {
   python make_trainset_statistics.py \
    --num_workers 5 --max_batches 50 \
    --data_root data/LibriSpeech_50h/wav_sel \
    --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
    --dataset PairWavDataset \
    --data_root data/ami \
    --data_cfg data/ami_data_ihm_sdm1357.cfg \
    --dataset AmiSegTupleWavDataset \
    --data_root . \
    --data_cfg data/swbd1_30k.cfg \
    --dataset PairWavDataset \
    --out_file data/libri_ami_swbd_stats.pkl
}

$chime5 && {
  python make_trainset_statistics.py \
    --num_workers 5 --max_batches 30 \
    --data_root /tmp-corpora/chime5segmented \
    --data_cfg data/chime5_seg_matched.cfg \
    --dataset AmiSegTupleWavDataset \
    --kaldi_root /disks/data1/pawel/repos/kaldi \
    --out_file data/chime5_seg_matched_kaldimfcc_plp_stats.pkl
}

$chime5_libri && {
  python make_trainset_statistics.py \
    --num_workers 5 --max_batches 50 \
    --data_root /tmp-corpora/chime5segmented \
    --data_cfg data/chime5_seg_matched.cfg \
    --dataset AmiSegTupleWavDataset \
    --data_root /tmp-corpora/LibriSpeech_50h/wav_sel \
    --data_cfg data/LibriSpeech_50h/librispeech_data_50h.cfg \
    --dataset LibriSpeechSegTupleWavDataset \
    --out_file data/chime5_libri_seg_matched_stats.pkl
}
