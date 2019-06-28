python -u nnet.py --spk2idx /export/corpora/mini_voxceleb/lists/utt2spk.npy --data_root /export/corpora/mini_voxceleb/train/ --train_guia /export/corpora/mini_voxceleb/lists/train_list \
       --log_freq 50 --batch_size 100 --lr 0.001 --save_path /home/monteiro/spkid_out/ \
       --model mlp --opt adam --patience 5 --train --lrdec 0.5 \
       --hidden_size 2048 --epoch 150 --sched_mode plateau \
       --fe_cfg ../cfg/PASE.cfg  \
       --fe_ckpt /export/fs01/monteijo/PASE.ckpt --seed 2 \
       --num_workers 3
