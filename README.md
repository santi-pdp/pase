### Data preparation

TODO

### Pre-trained model

PASE pre-trained parameters on LibriSpeech can be found <a href='veu.talp.cat/models/PASE.ckpt'>here</a>.

### Training

Train a PASE self-supervisedly with all active workers with command:

```
python -u train.py --batch_size 32 --epoch 100 --save_path pase_ckpt --num_workers 1 \
	--warmup 10000000 --net_cfg cfg/all.cfg --fe_cfg cfg/frontend_RF160ms_norm-emb100.cfg \
	--do_eval --data_cfg data/librispeech_data.cfg --min_lr 0.0005 --fe_lr 0.0005 \
	--data_root data/LibriSpeech/wavs/ --stats data/librispeech_stats.pkl --lrdec_step 30 --lrdecay 0.5
```

### Training Supervised Auxiliary Task

TODO

