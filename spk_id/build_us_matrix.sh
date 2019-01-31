#!/bin/bash

# Specify unsupervised ckpt folders as arguments 1 and 2

function print_usage {
	echo "Usage: $0 <unsupervised_ckpt> <ue_matrix_file>"
}

if [ $# -lt 2 ]; then
	echo "ERROR: Not enough input arguments!"
	print_usage
	exit 1
fi

U1_PATH="$1"
if [ ! -d $U1_PATH ]; then
	echo "Checkpoint path $U1_PATH NOT FOUND!"
	exit 2
fi

UE_MATRIX_FILE="$2"

# configure unsupervised epochs
UES="0 4 9 14 29 49"
# configure supervised epochs
SES="0 4 9 14 29 49"

# supervised model params
SUP_MODEL="mlp"
EPOCH=50
SCHED_MODE="step"
HIDDEN_SIZE=2048
LRDEC=0.5
SEED=4
# WARNING: SET THIS CORRECTLY
EMB_DIM=100
# TODO: --no-valid indicates to not make a validation partition
VALIDATION=""
# only used with validation
PATIENCE=5
OPT="adam"
BATCH_SIZE=100
LOG_FREQ=50
LR=0.001
RNN="--no-rnn"


if [ -f $UE_MATRIX_FILE ]; then
	echo "ERROR: Matrix $UE_MATRIX_FILE already exists!"
	exit 4
fi

for ue in $UES; do
	UE_BNAME=`basename $U1_PATH`
	SE_BNAME="$SUP_MODEL$HIDDEN_SIZE"_"$UE_BNAME"_FE$ue
	SAVE_PATH="$SAVE_ROOT/$SE_BNAME"
	FE_CKPT="$U1_PATH/FE_e$ue".ckpt
	echo $FE_CKPT 
	# train a supervised system per unsupervised ckpt in each of 2 ckpts
	python -u nnet.py --spk2idx $SPK2IDX --data_root $DATA_ROOT --train_guia $TRAIN_GUIA --log_freq $LOG_FREQ \
		--batch_size $BATCH_SIZE --lr $LR --save_path $SAVE_PATH --model $SUP_MODEL --opt $OPT --patience $PATIENCE \
		--train --lrdec $LRDEC --hidden_size $HIDDEN_SIZE --epoch $EPOCH \
		--sched_mode $SCHED_MODE $VALIDATION $RNN --emb_dim $EMB_DIM --fe_ckpt $FE_CKPT
	for se in $SES; do
		CKPT=$(python select_supervised_ckpt.py $SAVE_PATH $se)
		if [ ! -z "${CKPT##*[!0-9]*}" ] && [ $CKPT -ge 1 ]; then
			echo "File not found for UE $ue and SE $se"
			break
		fi
		LOG_FILE=`basename $CKPT`
		LOG_FILE=$SAVE_PATH/$LOG_FILE.log
		python -u nnet.py --spk2idx $SPK2IDX --data_root $DATA_ROOT --test_guia $TEST_GUIA \
			--test_ckpt $CKPT --model $SUP_MODEL --hidden_size $HIDDEN_SIZE --test \
			$RNN --emb_dim $EMB_DIM --test_log_file $LOG_FILE
		ACC=$(cat $LOG_FILE | grep "Test accuracy: " | perl -F: -alne 'print $F[1]' | sed 's/^\ //')
		echo -e "$ACC \c" >> $UE_MATRIX_FILE
	done
	sed -i 's/\ $//' $UE_MATRIX_FILE
	echo "" >> $UE_MATRIX_FILE
done

