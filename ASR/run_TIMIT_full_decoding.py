#
# To run a TIMIT experiment, go to the ASR folder and execute the following command:
#
# python run_TIMIT_full_decoding.py ../cfg/frontend/PASE+.cfg ../FE_e199.ckpt $SLURM_TMPDIR/TIMIT/ TIMIT_asr_exp cfg/MLP_PASE.cfg cfg/decoder.cfg 
#


# Importing libraries
import os
import sys
from neural_networks import MLP, context_window
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pickle
from pase.models.frontend import wf_builder
import soundfile as sf
import json

from data_io import write_mat, open_or_fd
from utils import run_shell

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu(trials=10):
    for j in range(trials):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2])
                            for x in open('tmp', 'r').readlines()]
        dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
        try:
            a = torch.rand(1).cuda(dev_)
            return dev_
        except:
            pass
            print('NO GPU AVAILABLE!!!')
            exit(1)

# Reading inputs
pase_cfg = sys.argv[1]  # e.g, '../cfg/frontend/PASE+.cfg'
pase_model = sys.argv[2]  # e.g, '../FE_e199.ckp' (download the pre-trained PASE+ model as described in the doc)
data_folder = sys.argv[3]  # e.g., '/home/mirco/Dataset/TIMIT'
output_folder = sys.argv[4]  # e.g., 'TIMIT_asr_exp'
cfg_file = sys.argv[5]  # e.g, cfg/MLP_pase.cfg
cfg_dec = sys.argv[6]  # e.g., cfg/decoder.cfg

skip_training = False

# using absolute path for output folder
output_folder=os.path.abspath(output_folder)

count_file = output_folder+'/count.npy'
ASR_model_file = output_folder+'/model.pkl'  # e.g., TIMIT_matconv_512/model.pkl

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not(skip_training):
    output_file = output_folder+'/res.res'
    output_model = output_folder+'/model.pkl'

    # Label files for TIMIT
    lab_file = 'TIMIT_lab_cd.pkl'
    lab_file_dev = 'TIMIT_lab_cd_dev.pkl'

    # File list for TIMIT
    tr_lst_file = 'timit_tr_kaldi.lst'
    dev_lst_file = 'timit_dev_kaldi.lst'

    tr_lst = [line.rstrip('\n') for line in open(tr_lst_file)]
    dev_lst = [line.rstrip('\n') for line in open(dev_lst_file)]

    # Training parameters

    with open(cfg_file, "r") as read_file:
        cfg = json.load(read_file)

    # Parameters
    N_epochs = int(cfg['N_epochs'])
    seed = int(cfg['seed'])
    batch_size = int(cfg['batch_size'])
    halving_factor = float(cfg['halving_factor'])
    lr = float(cfg['lr'])
    left = int(cfg['left']) # Reduce this to minimize memory (but it has an effect on performance too)
    right = int(cfg['right']) # Reduce this to minimize memory (but it has an effect on performance too)
    avg_spk = bool(cfg['avg_spk'])
    dnn_lay = cfg['dnn_lay']
    dnn_drop = cfg['dnn_drop']
    dnn_use_batchnorm = cfg['dnn_use_batchnorm']
    dnn_use_laynorm = cfg['dnn_use_laynorm']
    dnn_use_laynorm_inp = cfg['dnn_use_laynorm_inp']
    dnn_use_batchnorm_inp = cfg['dnn_use_batchnorm_inp']
    dnn_act = cfg['dnn_act']

    options = {}
    options['dnn_lay'] = dnn_lay
    options['dnn_drop'] = dnn_drop
    options['dnn_use_batchnorm'] = dnn_use_batchnorm
    options['dnn_use_laynorm'] = dnn_use_laynorm
    options['dnn_use_laynorm_inp'] = dnn_use_laynorm_inp
    options['dnn_use_batchnorm_inp'] = dnn_use_batchnorm_inp
    options['dnn_act'] = dnn_act

    device = get_freer_gpu()

    # folder creation
    text_file = open(output_file, "w")

    # Loading pase
    pase = wf_builder(pase_cfg)
    pase.load_pretrained(pase_model, load_last=True, verbose=False)
    pase.to(device)
    pase.eval()

    # reading the training signals
    print("Waveform reading...")
    fea = {}
    for wav_file in tr_lst:
        [signal, fs] = sf.read(data_folder+'/'+wav_file)
        signal = signal/np.max(np.abs(signal))
        signal = signal.astype(np.float32)

        fea_id = wav_file.split('/')[-2]+'_' + \
            wav_file.split('/')[-1].split('.')[0]
        fea[fea_id] = torch.from_numpy(
            signal).float().to(device).view(1, 1, -1)

    # reading the dev signals
    fea_dev = {}
    for wav_file in dev_lst:
        [signal, fs] = sf.read(data_folder+'/'+wav_file)
        signal = signal/np.max(np.abs(signal))
        fea_id = wav_file.split('/')[-2]+'_' + \
            wav_file.split('/')[-1].split('.')[0]
        fea_dev[fea_id] = torch.from_numpy(
            signal).float().to(device).view(1, 1, -1)

    # Computing pase features for training
    print('Computing PASE features...')
    fea_pase = {}
    mean_spk = {}
    std_spk = {}
    for snt_id in fea.keys():
        pase.eval()

        if avg_spk:
            fea_pase[snt_id] = pase(fea[snt_id], device).to('cpu').detach()
        else:
            fea_pase[snt_id] = pase(
                fea[snt_id], device, mode='avg_norm').to('cpu').detach()
        fea_pase[snt_id] = fea_pase[snt_id].view(
            fea_pase[snt_id].shape[1], fea_pase[snt_id].shape[2]).transpose(0, 1)
        spk_id = snt_id.split('_')[0]
        if spk_id not in mean_spk:
            mean_spk[spk_id] = []
            std_spk[spk_id] = []
        mean_spk[spk_id].append(torch.mean(fea_pase[snt_id], dim=0))
        std_spk[spk_id].append(torch.std(fea_pase[snt_id], dim=0))

    # compute per-speaker mean and variance
    if avg_spk:
        for spk_id in mean_spk.keys():
            mean_spk[spk_id] = torch.mean(torch.stack(mean_spk[spk_id]), dim=0)
            std_spk[spk_id] = torch.mean(torch.stack(std_spk[spk_id]), dim=0)

        # apply speaker normalization
        for snt_id in fea.keys():
            spk_id = snt_id.split('_')[0]
            # /std_spk[spk_id]
            fea_pase[snt_id] = (fea_pase[snt_id]-mean_spk[spk_id])

    inp_dim = fea_pase[snt_id].shape[1]*(left+right+1)

    # Computing pase features for test
    fea_pase_dev = {}
    mean_spk_dev = {}
    std_spk_dev = {}

    for snt_id in fea_dev.keys():

        if avg_spk:
            fea_pase_dev[snt_id] = pase(
                fea_dev[snt_id], device).to('cpu').detach()
        else:
            fea_pase_dev[snt_id] = pase(
                fea_dev[snt_id], device, mode='avg_norm').to('cpu').detach()

        fea_pase_dev[snt_id] = fea_pase_dev[snt_id].view(
            fea_pase_dev[snt_id].shape[1], fea_pase_dev[snt_id].shape[2]).transpose(0, 1)
        spk_id = snt_id.split('_')[0]
        if spk_id not in mean_spk_dev:
            mean_spk_dev[spk_id] = []
            std_spk_dev[spk_id] = []
        mean_spk_dev[spk_id].append(torch.mean(fea_pase_dev[snt_id], dim=0))
        std_spk_dev[spk_id].append(torch.std(fea_pase_dev[snt_id], dim=0))

    # compute per-speaker mean and variance
    if avg_spk:
        for spk_id in mean_spk_dev.keys():
            mean_spk_dev[spk_id] = torch.mean(
                torch.stack(mean_spk_dev[spk_id]), dim=0)
            std_spk_dev[spk_id] = torch.mean(
                torch.stack(std_spk_dev[spk_id]), dim=0)

        # apply speaker normalization
        for snt_id in fea_dev.keys():
            spk_id = snt_id.split('_')[0]
            # /std_spk_dev[spk_id]
            fea_pase_dev[snt_id] = (fea_pase_dev[snt_id]-mean_spk_dev[spk_id])

    # Label file reading
    with open(lab_file, 'rb') as handle:
        lab = pickle.load(handle)

    with open(lab_file_dev, 'rb') as handle:
        lab_dev = pickle.load(handle)

    # Network initialization
    nnet = MLP(options, inp_dim)

    nnet.to(device)

    cost = nn.NLLLoss()

    # Optimizer initialization
    optimizer = optim.SGD(nnet.parameters(), lr=lr, momentum=0.0)

    # Seeds initialization
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Batch creation (train)
    fea_lst = []
    lab_lst = []

    print("Data Preparation...")
    for snt in fea_pase.keys():
        if fea_pase[snt].shape[0]-lab[snt].shape[0] != 2:
            if fea_pase[snt].shape[0]-lab[snt].shape[0] == 3:
                fea_lst.append(fea_pase[snt][:-3])
                lab_lst.append(lab[snt])
            elif fea_pase[snt].shape[0]-lab[snt].shape[0] == 1:
                fea_lst.append(fea_pase[snt][:-1])
                lab_lst.append(lab[snt])
            else:
                print('length error')
                sys.exit(0)
        else:
            fea_lst.append(fea_pase[snt][:-2])
            lab_lst.append(lab[snt])

    # batch creation (dev)
    fea_lst_dev = []
    lab_lst_dev = []
    for snt in fea_pase_dev.keys():
        if fea_pase_dev[snt].shape[0]-lab_dev[snt].shape[0] != 2:
            if fea_pase_dev[snt].shape[0]-lab_dev[snt].shape[0] == 3:
                fea_lst_dev.append(fea_pase_dev[snt][:-3])
                lab_lst_dev.append(lab_dev[snt])
            elif fea_pase_dev[snt].shape[0]-lab_dev[snt].shape[0] == 1:
                fea_lst_dev.append(fea_pase_dev[snt][:-1])
                lab_lst_dev.append(lab_dev[snt])
            else:
                print('length error')
                sys.exit(0)
        else:

            fea_lst_dev.append(fea_pase_dev[snt][:-2])
            lab_lst_dev.append(lab_dev[snt])

    # feature matrix (training)
    fea_conc = np.concatenate(fea_lst)
    fea_conc = context_window(fea_conc, left, right)

    # feature matrix (dev)
    fea_conc_dev = np.concatenate(fea_lst_dev)
    fea_conc_dev = context_window(fea_conc_dev, left, right)

    # lab matrix
    lab_conc = np.concatenate(lab_lst)
    lab_conc_dev = np.concatenate(lab_lst_dev)

    if right > 0:
        lab_conc = lab_conc[left:-right]
        lab_conc_dev = lab_conc_dev[left:-right]
    else:
        lab_conc = lab_conc[left:]
        lab_conc_dev = lab_conc_dev[left:]

    # lab normalization
    lab_conc = lab_conc-lab_conc.min()

    # create count file (useful to scale posteriors before decoding)
    unique, counts = np.unique(lab_conc, return_counts=True)
    count_file = output_folder+'/'+'count.npy'
    np.save(count_file, counts)

    id_file = output_folder+'/'+'ids.npy'
    np.save(id_file, unique)

    lab_conc_dev = lab_conc_dev-lab_conc_dev.min()

    # dataset composition
    dataset = np.concatenate([fea_conc, lab_conc.reshape(-1, 1)], axis=1)
    dataset_dev = np.concatenate(
        [fea_conc_dev, lab_conc_dev.reshape(-1, 1)], axis=1)

    dataset_dev = torch.from_numpy(dataset_dev).float()

    # computing N_batches
    N_ex_tr = dataset.shape[0]
    N_batches = int(N_ex_tr/batch_size)

    N_ex_dev = dataset_dev.shape[0]
    N_batches_dev = int(N_ex_dev/batch_size)

    err_batch_history = []

    # Training loop
    print("Training...")
    for ep in range(N_epochs):
        err_batches = 0
        loss_batches = 0

        beg_batch = 0

        # training modality
        nnet.train()

        # random shuffling
        np.random.shuffle(dataset)

        for batch_id in range(N_batches):

            # Batch selection
            end_batch = beg_batch+batch_size
            batch = torch.from_numpy(dataset[beg_batch:end_batch]).float()
            batch = batch.to(device)

            fea_batch = batch[:, :-1]
            lab_batch = batch[:, -1].long()

            # computing the output probabilities
            out = nnet(fea_batch)

            # computing the loss
            loss = cost(out, lab_batch)

            # computing the error
            pred = torch.max(out, dim=1)[1]
            err = torch.mean((pred != lab_batch).float())

            # loss/error accumulation
            err_batches = err_batches+err.detach()
            loss_batches = loss_batches+loss.detach()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            beg_batch = end_batch

        # evaluation
        nnet.eval()
        beg_batch = 0

        err_batches_dev = 0
        loss_batches_dev = 0

        with torch.no_grad():
            for batch_id in range(N_batches_dev):

                end_batch = beg_batch+batch_size

                batch_dev = dataset_dev[beg_batch:end_batch]
                batch_dev = batch_dev.to(device)

                fea_batch_dev = batch_dev[:, :-1]
                lab_batch_dev = batch_dev[:, -1].long()

                out = nnet(fea_batch_dev)

                loss = cost(out, lab_batch_dev)

                pred = torch.max(out, dim=1)[1]
                err = torch.mean((pred != lab_batch_dev).float())

                err_batches_dev = err_batches_dev+err.detach()
                loss_batches_dev = loss_batches_dev+loss.detach()

                beg_batch = end_batch

        err_batch_history.append(err_batches_dev/N_batches_dev)

        print("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te=%f lr=%f" % (ep, loss_batches/N_batches,
            err_batches/N_batches, loss_batches_dev/N_batches_dev, err_batches_dev/N_batches_dev, lr))
        
        text_file = open(output_file, "a+")
        text_file.write("epoch=%i loss_tr=%f err_tr=%f loss_te=%f err_te=%f lr=%f\n" % (
            ep, loss_batches/N_batches, err_batches/N_batches, loss_batches_dev/N_batches_dev, err_batches_dev/N_batches_dev, lr))
        text_file.close()

        # learning rate annealing
        if ep > 0:

            # save the model if it is the best (according to err dev)
            if min(err_batch_history) == err_batch_history[-1]:
                torch.save(nnet.state_dict(), output_model)

            if (err_batch_history[-2]-err_batch_history[-1])/err_batch_history[-2] < 0.0025:
                lr = lr*halving_factor
                optimizer.param_groups[0]['lr'] = lr

    print('BEST ERR=%f' % (min(err_batch_history)))
    print('BEST ACC=%f' % (1-min(err_batch_history)))
    text_file = open(output_file, "a+")
    text_file.write('BEST_ERR=%f\n' % (min(err_batch_history)))
    text_file.write('BEST_ACC=%f\n' % (1-min(err_batch_history)))
    text_file.close()


ark_file = output_folder+'/post.ark'

device = 'cuda'

decoding_only = False

if not decoding_only:
    counts = np.load(count_file)
    log_counts = np.log(counts/np.sum(counts))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_file = output_folder+'/res.res'

    dev_lst_file = 'timit_te.lst'
    dev_lst = [line.rstrip('\n') for line in open(dev_lst_file)]

    # Training parameters
    with open(cfg_file, "r") as read_file:
        cfg = json.load(read_file)

    with open(cfg_dec, "r") as read_file:
        cfg_dec = json.load(read_file)

    with open(output_folder+'/dec_cfg.ini', 'w') as file:
        file.write('[decoding]\n')
        for key in cfg_dec.keys():
            file.write('%s=%s\n' % (key, cfg_dec[key]))

    # Parameters
    N_epochs = int(cfg['N_epochs'])
    seed = int(cfg['seed'])
    batch_size = int(cfg['batch_size'])
    halving_factor = float(cfg['halving_factor'])
    lr = float(cfg['lr'])
    left = int(cfg['left'])
    right = int(cfg['right'])
    avg_spk = bool(cfg['avg_spk'])
    dnn_lay = cfg['dnn_lay']
    dnn_drop = cfg['dnn_drop']
    dnn_use_batchnorm = cfg['dnn_use_batchnorm']
    dnn_use_laynorm = cfg['dnn_use_laynorm']
    dnn_use_laynorm_inp = cfg['dnn_use_laynorm_inp']
    dnn_use_batchnorm_inp = cfg['dnn_use_batchnorm_inp']
    dnn_act = cfg['dnn_act']

    options = {}
    options['dnn_lay'] = dnn_lay
    options['dnn_drop'] = dnn_drop
    options['dnn_use_batchnorm'] = dnn_use_batchnorm
    options['dnn_use_laynorm'] = dnn_use_laynorm
    options['dnn_use_laynorm_inp'] = dnn_use_laynorm_inp
    options['dnn_use_batchnorm_inp'] = dnn_use_batchnorm_inp
    options['dnn_act'] = dnn_act

    # folder creation
    #text_file = open(output_file, "w")

    # Loading pase
    pase = wf_builder(pase_cfg)
    pase.load_pretrained(pase_model, load_last=True, verbose=False)
    pase.to(device)
    pase.eval()

    # reading the training signals
    print("Waveform reading...")

    # reading the dev signals
    fea_dev = {}
    for wav_file in dev_lst:
        [signal, fs] = sf.read(data_folder+'/'+wav_file)
        signal = signal/np.max(np.abs(signal))
        fea_id = wav_file.split('/')[-2]+'_' + \
            wav_file.split('/')[-1].split('.')[0]
        fea_dev[fea_id] = torch.from_numpy(
            signal).float().to(device).view(1, 1, -1)

    # Computing pase features for training
    print('Computing PASE features...')

    # Computing pase features for test
    fea_pase_dev = {}
    mean_spk_dev = {}
    std_spk_dev = {}

    for snt_id in fea_dev.keys():

        if avg_spk:
            fea_pase_dev[snt_id] = pase(
                fea_dev[snt_id], device).to('cpu').detach()
        else:
            fea_pase_dev[snt_id] = pase(
                fea_dev[snt_id], device, mode='avg_norm').to('cpu').detach()

        fea_pase_dev[snt_id] = fea_pase_dev[snt_id].view(
            fea_pase_dev[snt_id].shape[1], fea_pase_dev[snt_id].shape[2]).transpose(0, 1)
        spk_id = snt_id.split('_')[0]
        if spk_id not in mean_spk_dev:
            mean_spk_dev[spk_id] = []
            std_spk_dev[spk_id] = []
        mean_spk_dev[spk_id].append(torch.mean(fea_pase_dev[snt_id], dim=0))
        std_spk_dev[spk_id].append(torch.std(fea_pase_dev[snt_id], dim=0))

    # compute per-speaker mean and variance
    if avg_spk:
        for spk_id in mean_spk_dev.keys():
            mean_spk_dev[spk_id] = torch.mean(
                torch.stack(mean_spk_dev[spk_id]), dim=0)
            std_spk_dev[spk_id] = torch.mean(
                torch.stack(std_spk_dev[spk_id]), dim=0)

        # apply speaker normalization
        for snt_id in fea_dev.keys():
            spk_id = snt_id.split('_')[0]
            # /std_spk_dev[spk_id]
            fea_pase_dev[snt_id] = (fea_pase_dev[snt_id]-mean_spk_dev[spk_id])
            fea_pase_dev[snt_id] = context_window(
                fea_pase_dev[snt_id], left, right)

    # Network initialization
    inp_dim = fea_pase_dev[snt_id].shape[1]
    nnet = MLP(options, inp_dim)
    nnet.to(device)

    nnet.load_state_dict(torch.load(ASR_model_file))
    nnet.eval()

    post_file = open_or_fd(ark_file, output_folder, 'wb')

    for snt_id in fea_dev.keys():
        pout = nnet(torch.from_numpy(fea_pase_dev[snt_id]).to(device).float())
        pout = pout-torch.tensor(log_counts).float().to(device)
        write_mat(output_folder, post_file, pout.data.cpu().numpy(), snt_id)

# doing decoding
print('Decoding...')
cmd_decode = cfg_dec['decoding_script_folder'] + '/' + cfg_dec['decoding_script'] + ' ' + \
    os.path.abspath(output_folder+'/dec_cfg.ini')+' ' + \
    output_folder+'/dec' + ' \"' + ark_file + '\"'
print(cmd_decode)
run_shell(cmd_decode)
