#!/bin/bash

libri=false
swbd=false
ami=false
libri_ami=false
libri_ami_sdm=false
libri_ami_swbd=false
chime5=true

$libri && {

python make_trainset_statistics.py \
  --data_root /tmp-corpora/LibriSpeech_50h/wav_sel \
  --data_cfg  /tmp-corpora/LibriSpeech_50h/librispeech_data_50h.cfg \
  --num_workers 5 \
  --out_file data/librispeech_tmp_stats.pkl

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
    --num_workers 5 --max_batches 50 \
    --data_root /tmp-corpora/chime5segmented \
    --data_cfg data/chime5_seg_matched.cfg \
    --dataset AmiSegTupleWavDataset \
    --out_file data/chime5_seg_matched_stats.pkl
}
