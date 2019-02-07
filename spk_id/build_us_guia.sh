#!/bin/bash

# build the guia file used to make US matrices (Unsup-Sup epoch exps)
# each output row means: [uepoch ckpt_path sepochs_to_test]

# configure unsupervised epochs
UES="4 9 19 39"
# configure supervised epochs
SES="9 19 39 79"

# Unsupervised MODELS guia
UMODELS=`cat ../ablation8_models.cfg`

OUT_FILE="ablation8_U.guia"

if [ -f $OUT_FILE ]; then
	rm -v $OUT_FILE
fi

for umodel in $UMODELS; do
	bname=`basename $umodel`
	bname=${bname%.*}
	# now we have the model name, build its repetitions
	# for the grid of U-S
	for ue in $UES; do
		echo "$ue $bname $SES" >> $OUT_FILE
	done
done

echo "Successfully created new $OUT_FILE"

