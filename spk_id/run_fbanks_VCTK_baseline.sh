#!/bin/bash

if [ $(hostname) == 'd5lnx26.upc.edu' ]; then
	SRUN='python'
else
	SRUN='srun -p veu -c1 --gres=gpu:1 --mem 15G python'
fi

SPK2IDX="../data/VCTK/spk_id/vctk_dict.npy"
DATA_ROOT="../data/VCTK/spk_id/fbanks"
TRAIN_GUIA="../data/VCTK/spk_id/vctk_tr.scp"
TEST_GUIA="../data/VCTK/spk_id/vctk_te.scp"
# root to store all supervised ckpts
SAVE_ROOT="US_vctk_ckpts/"

STATS="stats/vctk_fbanks.stats"
# 40 filter banks
ORDER=40

UE_MATRIX_FILE="FBANK_VCTK"

# supervised model params
SUP_MODEL="mlp"
EPOCH=100
SCHED_MODE="plateau"
HIDDEN_SIZE=2048
LRDEC=0.5
SEED=4
# only used with validation
PATIENCE=5
OPT="adam"
BATCH_SIZE=100
LOG_FREQ=50
LR=0.001

SAVE_PATH="$SAVE_ROOT/FBANK_MFCC_100epoch"


python -u mfcc_baseline.py --spk2idx $SPK2IDX --data_root $DATA_ROOT --train_guia $TRAIN_GUIA \
	--log_freq $LOG_FREQ --batch_size $BATCH_SIZE --lr $LR --save_path $SAVE_PATH \
	--model $SUP_MODEL --opt $OPT --patience $PATIENCE --train --lrdec $LRDEC \
	--hidden_size $HIDDEN_SIZE --epoch $EPOCH --sched_mode $SCHED_MODE \
	--seed $SEED $FT_FE --stats $STATS --ext fb.npy --np-fmt --order $ORDER

CKPT=$(python select_best_supervised_ckpt.py $SAVE_PATH)
echo $CKPT
if [ ! -z "${CKPT##*[!0-9]*}" ] && [ $CKPT -ge 1 ]; then
	echo "File not found for SE $se"
	break
fi

LOG_FILE=`basename $CKPT`
LOG_FILE=${LOG_FILE%.*}
LOG_FILE=$SAVE_PATH/$LOG_FILE.log
python -u mfcc_baseline.py --spk2idx $SPK2IDX --data_root $DATA_ROOT --test_guia $TEST_GUIA \
	--test_ckpt $CKPT --model $SUP_MODEL --hidden_size $HIDDEN_SIZE --test \
	--test_log_file $LOG_FILE --stats $STATS --ext fb.npy --np-fmt --order $ORDER

ACC=$(cat $LOG_FILE | grep "Test accuracy: " | perl -F: -alne 'print $F[1]' | sed 's/^\ //')
echo -e "$ACC \c" >> $UE_MATRIX_FILE
sed -i 's/\ $//' $UE_MATRIX_FILE
echo "" >> $UE_MATRIX_FILE
