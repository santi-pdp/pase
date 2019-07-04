#!/bin/bash

CUDA_VISIBLE_DEVICES=`/home/jtrmal/free-gpu` /export/fs01/pascual/miniconda3/bin/python -u run_IEMOCAP_fast.py $cfg $model /export/corpora/pase_data/IEMOCAP_ahsn_leave-two-speaker-out $save_path/iemocap-aux_$iteration
