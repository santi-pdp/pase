### Data preparation

Execute the data configuration to generate the description of the VCTK dataset:

```
python data_cfg_vctk.py --cfg_file data/vctk_data.cfg ~/DB/VCTK
```

Train a minionet model with command:

```
python train.py --rndmin_train --batch_size 16 --save_path ckpt_bsz16
```

`--rndmin_train` picks a random minion to backprop each turn, and this
trains a hardcoded architecture so far.
