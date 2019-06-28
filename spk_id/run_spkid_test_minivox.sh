python -u nnet.py --spk2idx /export/corpora/mini_voxceleb/lists/utt2spk.npy --data_root /export/corpora/mini_voxceleb/test/ --test_guia /export/corpora/mini_voxceleb/lists/test_list \
       --test_ckpt /home/monteiro/spkid_out/weights_MLP-MLP-15.ckpt --model mlp --hidden_size 2048 --test \
       --fe_cfg ../cfg/PASE.cfg --test_log_file out_ep15_mlp
