#!/usr/bin/bash

python -u make_trainset_statistics.py --data_cfg data/LibriSpeech_50h/LibriSpeech_50h/librispeech_data_50h.cfg --data_root data/LibriSpeech_50h/LibriSpeech_50h/wav_sel --kaldi_root ~/jsalt/kaldi/ --out_file data/libri_der2_jianyuan.pkl --dataset LibriSpeechSegTupleWavDataset --net_cfg cfg/workers_best_matconv.cfg --chunk_size 32000
