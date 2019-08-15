# Mirco Ravanelli
# Mila, June 2019

# This script runs a simple speaker recognition experiment on the top of PASE features. 
# The results are reported in terms of Frame Error Rate /Sentence Error Rates.
# This system is not designed for an extensive evaluation of PASE features, but mainly for quickly monitoring the performance of PASE during the self-supervised training phase.
# The results are printed in standard output and within the text file specified in the last argument.

# To run the speaker recognition exp on minivox celeb:
# python run_minivox_fast.py ../cfg/PASE.cfg ../PASE.ckpt /scratch/ravanelm/datasets/minivoxceleb_40spk/ minivoxceleb_40spk/minivox_tr_list.txt minivoxceleb_40spk/minivox_test_list.txt  minivoxceleb_40spk/utt2spk.npy  minivox_fina.res

# To run the language id experiment on minivoxforge:
# python run_minivox_fast.py ../cfg/PASE.cfg ../PASE.ckpt /scratch/ravanelm/datasets/mini_voxforge/ minivoxforge_tr_list.txt minivoxforge_test_list.txt  utt2lang.npy  minivoxforge.res

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import json
from neural_networks import MLP,context_window
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from pase.models.frontend import wf_builder
# from waveminionet.models.frontend import wf_builder #old models
import soundfile as sf
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

def get_nspk(utt2spk):
    lab_list = []
    for utt, spk in utt2spk.items():
        lab_list.append(int(spk))

    return max(lab_list)+1


pase_cfg=sys.argv[1] # e.g, '../cfg/PASE.cfg'
pase_model=sys.argv[2] # e.g, '../PASE.ckpt'
data_folder=sys.argv[3] # eg. '/home/mirco/Dataset/mini_voxceleb minivox'
tr_lst_file=sys.argv[4] # e.g., 'minivox_tr_list.txt'
dev_lst_file=sys.argv[5] # e.g., 'minvox_test_list.txt'
lab_file=sys.argv[6] # e.g., 'minvox_test_list.txt'
output_file=sys.argv[7] # e.g., 


lab=np.load(lab_file, allow_pickle=True).item()

# get number of speakers

nspk=get_nspk(lab)


tr_lst = [line.rstrip('\n') for line in open(tr_lst_file)]
dev_lst = [line.rstrip('\n') for line in open(dev_lst_file)]

# Training parameters
N_epochs=24
seed=1234
batch_size=128
halving_factor=0.5
lr=0.001
left=0
right=0

# Neural network parameters
options={}
options['dnn_lay']='256,'+str(nspk)
options['dnn_drop']='0.15,0.0'
options['dnn_use_batchnorm']='False,False'
options['dnn_use_laynorm']='True,False'
options['dnn_use_laynorm_inp']='True'
options['dnn_use_batchnorm_inp']='False'
options['dnn_act']='relu,softmax'

device=0 #get_freer_gpu()


# output file creation
text_file=open(output_file, "w")

# Loading pase
pase = wf_builder(pase_cfg)
pase.load_pretrained(pase_model, load_last=True, verbose=False)
pase.to(device)
pase.eval()

# reading the training signals
print("Waveform reading...")
fea={}
for wav_file in tr_lst:
    [signal, fs] = sf.read(data_folder+'/train/'+wav_file)
    signal=signal/np.max(np.abs(signal))
    signal = signal.astype(np.float32)
    
    fea_id=wav_file.split('/')[-1]
    fea[fea_id]=torch.from_numpy(signal).float().to(device).view(1,1,-1)


# reading the dev signals
fea_dev={}
for wav_file in dev_lst:
    [signal, fs] = sf.read(data_folder+'/test/'+wav_file)
    signal=signal/np.max(np.abs(signal))
    fea_id=wav_file.split('/')[-1]
    fea_dev[fea_id]=torch.from_numpy(signal).float().to(device).view(1,1,-1)



# Computing pase features for training
print('Computing PASE features...')
fea_pase={}
for snt_id in fea.keys():
    pase.eval()
    fea_pase[snt_id] = pase(fea[snt_id], device, mode='avg_concat').to('cpu').detach()
    fea_pase[snt_id]=fea_pase[snt_id].view(fea_pase[snt_id].shape[1],fea_pase[snt_id].shape[2]).transpose(0,1)

inp_dim=fea_pase[snt_id].shape[1]*(left+right+1)

# Computing pase features for test
fea_pase_dev={}
for snt_id in fea_dev.keys():
    fea_pase_dev[snt_id] = pase(fea_dev[snt_id], device,mode='avg_concat').detach()
    # fea_pase_dev[snt_id]=pase(fea_dev[snt_id]).detach()
    fea_pase_dev[snt_id]=fea_pase_dev[snt_id].view(fea_pase_dev[snt_id].shape[1],fea_pase_dev[snt_id].shape[2]).transpose(0,1)


  

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
for snt in fea_pase.keys():
        fea_lst.append(fea_pase[snt])
        lab_lst.append(np.zeros(fea_pase[snt].shape[0])+lab[snt])

    
# feature matrix (training)
fea_conc=np.concatenate(fea_lst)
fea_conc=context_window(fea_conc,left,right)

# feature normalization
mean=np.mean(fea_conc,axis=0)
std=np.std(fea_conc,axis=0)

# normalization

fea_conc=(fea_conc-mean)/std

mean=torch.from_numpy(mean).float().to(device)
std=torch.from_numpy(std).float().to(device)

# lab matrix
lab_conc=np.concatenate(lab_lst)

if right>0:
    lab_conc=lab_conc[left:-right]
else:
    lab_conc=lab_conc[left:]


# dataset composition
dataset=np.concatenate([fea_conc,lab_conc.reshape(-1,1)],axis=1)

# shuffling
np.random.shuffle(dataset)

dataset=torch.from_numpy(dataset).float().to(device)

# computing N_batches
N_ex_tr=dataset.shape[0]
N_batches=int(N_ex_tr/batch_size)


err_dev_fr_history=[]
err_dev_snt_history=[]

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
    
    
    with torch.no_grad():
    
        err_dev_fr_mean=0
        err_dev_snt_mean=0
        loss_dev_mean=0
        
        N_dev_snt=len(list(fea_pase_dev.keys()))
        
        for dev_snt in fea_pase_dev.keys():

             fea_dev_norm=(fea_pase_dev[dev_snt]-mean)/std
             out_dev=nnet(fea_dev_norm)
             lab_snt=torch.zeros(fea_pase_dev[dev_snt].shape[0])+lab[dev_snt]
             lab_snt=lab_snt.long().to(device)
             loss_dev=cost(out_dev,lab_snt)
             
             # frame level error
             pred_dev=torch.max(out_dev,dim=1)[1] 
             err_dev = torch.mean((pred_dev!=lab_snt).float())
             
             # sentence error level
             prob_sum=torch.sum(out_dev,dim=0)
             pred_dev_snt=torch.argmax(prob_sum) 
             err_snt=(pred_dev_snt!=lab_snt[0]).float()
             
             err_dev_fr_mean=err_dev_fr_mean+err_dev.detach()
             loss_dev_mean=loss_dev_mean+loss_dev.detach()
             err_dev_snt_mean=err_dev_snt_mean+err_snt.detach()
         
         
    err_dev_fr_history.append(err_dev_fr_mean/N_dev_snt)
    err_dev_snt_history.append(err_dev_snt_mean/N_dev_snt)
    
    
    print("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te_fr=%f err_te_snt=%f lr=%f" %(ep,loss_batches/N_batches,err_batches/N_batches,loss_dev_mean/N_dev_snt,err_dev_fr_mean/N_dev_snt,err_dev_snt_mean/N_dev_snt,lr))
    text_file.write("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te_fr=%f err_te_snt=%f lr=%f \n" %(ep,loss_batches/N_batches,err_batches/N_batches,loss_dev_mean/N_dev_snt,err_dev_fr_mean/N_dev_snt,err_dev_snt_mean/N_dev_snt,lr))
    
    # learning rate annealing
    if ep>0:
        if (err_dev_fr_history[-2]-err_dev_fr_history[-1])/err_dev_fr_history[-2]<0.0025:
            lr=lr*halving_factor
            optimizer.param_groups[0]['lr']=lr


print('BEST ERR=%f' %(min(err_dev_fr_history)))
print('BEST ACC=%f' %(1-min(err_dev_fr_history)))
text_file.write('BEST_ERR=%f\n' %(min(err_dev_fr_history)))
text_file.write('BEST_ACC=%f\n' %(1-min(err_dev_fr_history)))
text_file.close()
    
    
    
    



