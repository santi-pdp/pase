#!/bin/bash

stage=3

data_root=/export/corpora/ami
out_root=/export/team-mic/corpora/ami

if [ $stage -le 1 ]; then

  python prepare_segmented_dataset_ami.py \
    --data_root $data_root \
    --out_root $out_root \
    --ami_meeting_ids ami_split_train.list \
    --map_ihm2sdm 1,3,5,7

  python prepare_segmented_dataset_ami.py \
    --data_root $data_root \
    --out_root $out_root \
    --ami_meeting_ids ami_split_valid.list \
    --map_ihm2sdm 1,3,5,7

  find $out_root -iname '*.wav' > ami_all.list
  grep -f ami_split_train.list ami_all.list > ami_train.scp
  grep -f ami_split_valid.list ami_all.list > ami_test.scp

fi


if [ $stage -le 2 ]; then
  python unsupervised_data_cfg_ami.py \
   --data_root $out_root \
   --train_scp ami_train.scp \
   --test_scp ami_test.scp \
   --map_ihm2sdm 1,3,5,7 \
   --cfg_file ami_data_ihm_sdm1357.cfg
fi

if [ $stage -le 3 ]; then

  python ../../make_trainset_statistics.py \
    --data_root $out_root \
    --data_cfg ami_data_ihm_sdm1357.cfg \
    --exclude_keys 'chunk_rand', 'chunk_ctxt'
    --out_file ami_ihm_sdm_stats.pkl

  python ../../make_trainset_statistics.py \
    --data_root $out_root \
    --data_cfg ami_data_ihm_sdm1357.cfg \
    --exclude_keys 'chunk', 'chunk_rand', 'chunk_ctxt'
    --out_file ami_ihm_stats.pkl

fi
