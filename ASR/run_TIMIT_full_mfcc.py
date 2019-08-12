# Mirco Ravanelli
# Mila, June 2019

# This script runs a simple speech recognition experiment on the top of MFCC features. 
# The results are reported in terms of Frame Error Rate over phonemes (context-independent). 
# This system is not designed for an extensive evaluation of MFCC features, but mainly for quickly monitoring the performance of MFCC during the self-supervised training phase.
# The results are printed in standard output and within the text file specified in the last argument.

# To run it:
# python run_TIMIT_fast.py  /home/mirco/Dataset/TIMIT  TIMIT_asr_exp.res
#
# To run the experiment with the noisy and reverberated version of TIMIT, just change the data folder with the one containing TIMIT_rev_noise.

import librosa

import os
import sys
from neural_networks import MLP,context_window
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from pase.models.frontend import wf_builder
# from waveminionet.models.frontend import wf_builder #old models
import soundfile as sf
import os
import json
# import pase.models as models
# import models.WorkerScheduler
from pase.models.WorkerScheduler.encoder import *

def get_freer_gpu(trials=10):
    for j in range(trials):
         os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
         memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
         dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
         try:
            a = torch.rand(1).cuda(dev_)
            return dev_
         except: 
            pass
            print('NO GPU AVAILABLE!!!')
            exit(1)


data_folder=sys.argv[1] # e.g., '/home/mirco/Dataset/TIMIT'
output_file=sys.argv[2] # e.g., 'TIMIT_asr_exp.res'


# Label files for TIMIT
lab_file='TIMIT_lab_cd.pkl'
lab_file_dev='TIMIT_lab_cd_dev.pkl'

# File list for TIMIT
tr_lst_file='timit_tr.lst'
dev_lst_file='timit_dev.lst'

tr_lst = [line.rstrip('\n') for line in open(tr_lst_file)]

dev_lst = [line.rstrip('\n') for line in open(dev_lst_file)]

# Training parameters
N_epochs=24
seed=1234
batch_size=128
halving_factor=0.5
lr=0.08
left=8
right=8

# Neural network parameters
options={}
options['dnn_lay']='1024,1024,1024,1024,1024,1973'
options['dnn_drop']='0.15,0.15,0.15,0.15,0.15,0.0'
options['dnn_use_batchnorm']='True,True,True,True,True,False'
options['dnn_use_laynorm']='False,False,False,False,False,False'
options['dnn_use_laynorm_inp']='False'
options['dnn_use_batchnorm_inp']='False'
options['dnn_act']='relu,relu,relu,relu,relu,softmax'



device=get_freer_gpu()


# folder creation
text_file=open(output_file, "w")


# reading the training signals
print("Waveform reading...")
fea={}
for wav_file in tr_lst:
    # sound file
    [signal, fs] = sf.read(data_folder+'/'+wav_file)
    signal=signal/np.max(np.abs(signal))
    signal = signal.astype(np.float32)
    # Librosa loader (same results, but much slower)
    #signal, sr = librosa.load(data_folder+'/'+wav_file,sr=None)

    fea_id=wav_file.split('/')[-2]+'_'+wav_file.split('/')[-1].split('.')[0]
    fea[fea_id]=signal


# reading the dev signals
fea_dev={}
for wav_file in dev_lst:
    [signal, fs] = sf.read(data_folder+'/'+wav_file)
    signal=signal/np.max(np.abs(signal))
    signal = signal.astype(np.float32)
    #signal, sr = librosa.load(data_folder+'/'+wav_file,sr=None)
    fea_id=wav_file.split('/')[-2]+'_'+wav_file.split('/')[-1].split('.')[0]
    fea_dev[fea_id]=signal


# Computing mfcc
print('Computing MFCC features...')
fea_mfcc={}
for snt_id in fea.keys():
    mfcc_stat=librosa.feature.mfcc(y=fea[snt_id], sr=16000,n_mfcc=13,hop_length=160,htk=True)
    mfcc_delta = librosa.feature.delta(mfcc_stat,order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc_stat,order=2)

    fea_mfcc[snt_id]=np.concatenate([mfcc_stat,mfcc_delta,mfcc_delta_delta]).transpose(1,0)
    


inp_dim=fea_mfcc[snt_id].shape[1]*(left+right+1)

# Computing mfcc for test
fea_mfcc_dev={}
for snt_id in fea_dev.keys():
    mfcc_stat=librosa.feature.mfcc(y=fea_dev[snt_id], sr=16000,n_mfcc=13,hop_length=160,htk=True)
    mfcc_delta = librosa.feature.delta(mfcc_stat,order=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc_stat,order=2)

    fea_mfcc_dev[snt_id]=np.concatenate([mfcc_stat,mfcc_delta,mfcc_delta_delta]).transpose(1,0)
    


  
# Label file reading
with open(lab_file, 'rb') as handle:
    lab = pickle.load(handle)

with open(lab_file_dev, 'rb') as handle:
    lab_dev = pickle.load(handle)
    

# Network initialization
nnet=MLP(options,inp_dim)

nnet.to(device)

cost=nn.NLLLoss()

# Optimizer initialization
optimizer = optim.SGD(nnet.parameters(), lr=lr, momentum=0.0)

# Seeds initialization
np.random.seed(seed)
torch.manual_seed(seed)

# Batch creation (train)
fea_lst=[]
lab_lst=[]

print("Data Preparation...")
for snt in fea_mfcc.keys():
    if fea_mfcc[snt].shape[0]-lab[snt].shape[0]!=2:
        if fea_mfcc[snt].shape[0]-lab[snt].shape[0]==3:
            fea_lst.append(fea_mfcc[snt][:-3])
            lab_lst.append(lab[snt])
        elif fea_mfcc[snt].shape[0]-lab[snt].shape[0]==1:
            fea_lst.append(fea_mfcc[snt][:-1])
            lab_lst.append(lab[snt])
        else:
            print('length error')
            sys.exit(0)
    else:
        fea_lst.append(fea_mfcc[snt][:-2])
        lab_lst.append(lab[snt])

# batch creation (dev)
fea_lst_dev=[]
lab_lst_dev=[]
for snt in fea_mfcc_dev.keys():
    if fea_mfcc_dev[snt].shape[0]-lab_dev[snt].shape[0]!=2:
        if fea_mfcc_dev[snt].shape[0]-lab_dev[snt].shape[0]==3:
            fea_lst_dev.append(fea_mfcc_dev[snt][:-3])
            lab_lst_dev.append(lab_dev[snt])
        elif fea_mfcc_dev[snt].shape[0]-lab_dev[snt].shape[0]==1:
            fea_lst_dev.append(fea_mfcc_dev[snt][:-1])
            lab_lst_dev.append(lab_dev[snt])
        else:
            print('length error')
            sys.exit(0)
    else:

        fea_lst_dev.append(fea_mfcc_dev[snt][:-2])
        lab_lst_dev.append(lab_dev[snt])
    
    

# feature matrix (training)
fea_conc=np.concatenate(fea_lst)
fea_conc=context_window(fea_conc,left,right)

# feature matrix (dev)
fea_conc_dev=np.concatenate(fea_lst_dev)
fea_conc_dev=context_window(fea_conc_dev,left,right)

# feature normalization
fea_conc=(fea_conc-np.mean(fea_conc,axis=0))/np.std(fea_conc,axis=0)
fea_conc_dev=(fea_conc_dev-np.mean(fea_conc_dev,axis=0))/np.std(fea_conc_dev,axis=0)


# lab matrix
lab_conc=np.concatenate(lab_lst)
lab_conc_dev=np.concatenate(lab_lst_dev)

if right>0:
    lab_conc=lab_conc[left:-right]
    lab_conc_dev=lab_conc_dev[left:-right]
else:
    lab_conc=lab_conc[left:]
    lab_conc_dev=lab_conc_dev[left:]

# lab normalization
lab_conc=lab_conc-lab_conc.min()
lab_conc_dev=lab_conc_dev-lab_conc_dev.min()


# dataset composition
dataset=np.concatenate([fea_conc,lab_conc.reshape(-1,1)],axis=1)
dataset_dev=np.concatenate([fea_conc_dev,lab_conc_dev.reshape(-1,1)],axis=1)

# shuffling
np.random.shuffle(dataset)

# converting to pytorch
#dataset=torch.from_numpy(dataset).float().to(device)
dataset=torch.from_numpy(dataset).float()
#dataset_dev=torch.from_numpy(dataset_dev).float().to(device)
dataset_dev=torch.from_numpy(dataset_dev).float()


# computing N_batches
N_ex_tr=dataset.shape[0]
N_batches=int(N_ex_tr/batch_size)

N_ex_dev=dataset_dev.shape[0]
N_batches_dev=int(N_ex_dev/batch_size)

err_batch_history=[]

# Training loop
print("Training...")
for ep in range(N_epochs):
    err_batches=0
    loss_batches=0
    
    beg_batch=0
    
    # training modality
    nnet.train()
    
    # random shuffling
    shuffle_index=torch.randperm(dataset.shape[0])
    dataset=dataset[shuffle_index]
    
    for batch_id in range(N_batches):
        
        # Batch selection
        end_batch=beg_batch+batch_size
        batch=dataset[beg_batch:end_batch]
        batch=batch.to(device)        
        fea_batch=batch[:,:-1]
        lab_batch=batch[:,-1].long()
        
        # computing the output probabilities
        out=nnet(fea_batch)
           
        # computing the loss
        loss=cost(out,lab_batch)
        
        # computing the error
        pred=torch.max(out,dim=1)[1] 
        err = torch.mean((pred!=lab_batch).float())

        # loss/error accumulation        
        err_batches=err_batches+err.detach()
        loss_batches=loss_batches+loss.detach()
    
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        beg_batch=end_batch
        

        
    # evaluation
    nnet.eval()
    beg_batch=0
    
    err_batches_dev=0
    loss_batches_dev=0
    
    with torch.no_grad():
        for batch_id in range(N_batches_dev):
            
            end_batch=beg_batch+batch_size
            
            batch_dev=dataset_dev[beg_batch:end_batch]
            batch_dev=batch_dev.to(device)
            
            fea_batch_dev=batch_dev[:,:-1]
            lab_batch_dev=batch_dev[:,-1].long()
            
            out=nnet(fea_batch_dev)
            
            loss=cost(out,lab_batch_dev)
            
            pred=torch.max(out,dim=1)[1] 
            err = torch.mean((pred!=lab_batch_dev).float())
            
            err_batches_dev=err_batches_dev+err.detach()
            loss_batches_dev=loss_batches_dev+loss.detach()
            
            beg_batch=end_batch
        
    
    err_batch_history.append(err_batches_dev/N_batches_dev)
    
    
    print("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te=%f lr=%f" %(ep,loss_batches/N_batches,err_batches/N_batches,loss_batches_dev/N_batches_dev,err_batches_dev/N_batches_dev,lr))
    text_file.write("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te=%f lr=%f\n" %(ep,loss_batches/N_batches,err_batches/N_batches,loss_batches_dev/N_batches_dev,err_batches_dev/N_batches_dev,lr))

    # learning rate annealing
    if ep>0:
        if (err_batch_history[-2]-err_batch_history[-1])/err_batch_history[-2]<0.0025:
            lr=lr*halving_factor
            optimizer.param_groups[0]['lr']=lr


print('BEST ERR=%f' %(min(err_batch_history)))
print('BEST ACC=%f' %(1-min(err_batch_history)))
text_file.write('BEST_ERR=%f\n' %(min(err_batch_history)))
text_file.write('BEST_ACC=%f\n' %(1-min(err_batch_history)))
text_file.close()
    
    
    
    



