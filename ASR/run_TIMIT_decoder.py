# Mirco Ravanelli
# Mila, June 2019

import torch
import json
from neural_networks import MLP,context_window
import os
from pase.models.frontend import wf_builder
import soundfile as sf
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_io import write_mat,open_or_fd
from utils import run_shell
import sys

model_file='/home/mirco/pase/ASR/TIMIT_out_exp/model.pkl'
cfg_file='cfg/MLP_PASE.cfg'
cfg_pase='/home/mirco/pase/pase_models/MATconv_512_160ep/new/PASE_MAT_sinc_jiany_512.cfg'
cfg_dec='cfg/decoder.cfg' 
pase_ckpt='/home/mirco/pase/pase_models/MATconv_512_160ep/new/FE_e99.ckpt' 
data_folder='/home/mirco/Dataset/TIMIT'
output_folder='/home/mirco/pase/ASR/TIMIT_out_exp'
count_file='/home/mirco/pase/ASR/TIMIT_out_exp/count.npy'

ark_file=output_folder+'/post.ark'

device='cuda'

decoding_only=False

if not decoding_only:
    counts=np.load(count_file)
    
    log_counts=np.log(np.sum(counts)/np.sum(counts))
       
    
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    output_file=output_folder+'/res.res'
    
    dev_lst_file='timit_dev.lst'
    dev_lst = [line.rstrip('\n') for line in open(dev_lst_file)]
    
    
    # Training parameters
    with open(cfg_file, "r") as read_file:
        cfg = json.load(read_file)
     
    with open(cfg_dec, "r") as read_file:
        cfg_dec = json.load(read_file)
    
    with open(output_folder+'/dec_cfg.ini', 'w') as file:  
        file.write('[decoding]\n')
        for key in cfg_dec.keys():
                file.write('%s=%s\n' %(key,cfg_dec[key]))
    
    
    
    # Parameters
    N_epochs=int(cfg['N_epochs'])
    seed=int(cfg['seed'])
    batch_size=int(cfg['batch_size'])
    halving_factor=float(cfg['halving_factor'])
    lr=float(cfg['lr'])
    left=int(cfg['left'])
    right=int(cfg['right'])
    avg_spk=bool(cfg['avg_spk'])
    dnn_lay=cfg['dnn_lay']
    dnn_drop=cfg['dnn_drop']
    dnn_use_batchnorm=cfg['dnn_use_batchnorm']
    dnn_use_laynorm=cfg['dnn_use_laynorm']
    dnn_use_laynorm_inp=cfg['dnn_use_laynorm_inp']
    dnn_use_batchnorm_inp=cfg['dnn_use_batchnorm_inp']
    dnn_act=cfg['dnn_act']
    
    
    options={}
    options['dnn_lay']=dnn_lay
    options['dnn_drop']=dnn_drop
    options['dnn_use_batchnorm']=dnn_use_batchnorm
    options['dnn_use_laynorm']=dnn_use_laynorm
    options['dnn_use_laynorm_inp']=dnn_use_laynorm_inp
    options['dnn_use_batchnorm_inp']=dnn_use_batchnorm_inp
    options['dnn_act']=dnn_act
    
    
    
    # folder creation
    text_file=open(output_file, "w")
    
    # Loading pase
    pase =wf_builder(cfg_pase)
    pase.load_pretrained(pase_ckpt, load_last=True, verbose=False)
    pase.to(device)
    pase.eval()
    
    # reading the training signals
    print("Waveform reading...")
    
    # reading the dev signals
    fea_dev={}
    for wav_file in dev_lst:
        [signal, fs] = sf.read(data_folder+'/'+wav_file)
        signal=signal/np.max(np.abs(signal))
        fea_id=wav_file.split('/')[-2]+'_'+wav_file.split('/')[-1].split('.')[0]
        fea_dev[fea_id]=torch.from_numpy(signal).float().to(device).view(1,1,-1)
    
    # Computing pase features for training
    print('Computing PASE features...')
    
    
    # Computing pase features for test
    fea_pase_dev={}
    mean_spk_dev={}
    std_spk_dev={}
    
    for snt_id in fea_dev.keys():
    
        if avg_spk:
            fea_pase_dev[snt_id]=pase(fea_dev[snt_id], device).to('cpu').detach()
        else:
            fea_pase_dev[snt_id]=pase(fea_dev[snt_id], device, mode='avg_norm').to('cpu').detach()
    
        fea_pase_dev[snt_id]=fea_pase_dev[snt_id].view(fea_pase_dev[snt_id].shape[1],fea_pase_dev[snt_id].shape[2]).transpose(0,1)
        spk_id=snt_id.split('_')[0]
        if spk_id not in mean_spk_dev:
            mean_spk_dev[spk_id]=[]
            std_spk_dev[spk_id]=[]
        mean_spk_dev[spk_id].append(torch.mean(fea_pase_dev[snt_id],dim=0))
        std_spk_dev[spk_id].append(torch.std(fea_pase_dev[snt_id],dim=0))
    
    # compute per-speaker mean and variance
    if avg_spk:
        for spk_id in mean_spk_dev.keys():
            mean_spk_dev[spk_id]=torch.mean(torch.stack(mean_spk_dev[spk_id]),dim=0)
            std_spk_dev[spk_id]=torch.mean(torch.stack(std_spk_dev[spk_id]),dim=0)
    
        # apply speaker normalization
        for snt_id in fea_dev.keys():
            spk_id=snt_id.split('_')[0]
            fea_pase_dev[snt_id]=(fea_pase_dev[snt_id]-mean_spk_dev[spk_id])#/std_spk_dev[spk_id]
            fea_pase_dev[snt_id]=context_window(fea_pase_dev[snt_id],left,right)
    
    
    # Network initialization
    inp_dim=fea_pase_dev[snt_id].shape[1]*(left+right+1)
    nnet=MLP(options,inp_dim)
    nnet.to(device)
    
    nnet.load_state_dict(torch.load(model_file))
    nnet.eval()
    
    post_file=open_or_fd(ark_file,output_folder,'wb')
    
    for snt_id in fea_dev.keys():
         pout=nnet(torch.from_numpy(fea_pase_dev[snt_id]).to(device).float())
         # TO DO IT!!
         #pout=pout-log_counts
         write_mat(output_folder,post_file, pout.data.cpu().numpy(), snt_id)

# doing decoding
print('Decoding...')
cmd_decode=cfg_dec['decoding_script_folder'] +'/'+ cfg_dec['decoding_script']+ ' '+os.path.abspath(output_folder+'/dec_cfg.ini')+' '+ output_folder+'/dec' + ' \"'+ ark_file + '\"'      
run_shell(cmd_decode)
 

    



