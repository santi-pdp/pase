import json
import librosa
import argparse
import random
from random import shuffle
import numpy as np
import os
import re

def get_file_dur(fname):
    x, rate = librosa.load(fname, sr=None)
    return len(x)

def parse_list(file_in, opts):

    def utt2spk_fun(path):
        bsn = os.path.basename(path)
        match = re.match(r'(.*Headset\-\d).*', bsn)
        spk = None
        if match:
            spk = match.group(1)
        return bsn, spk

    def sdm2ihm_and_chan_fun(path):
        bsn = os.path.basename(path)
        match = re.match(r'(.*Headset\-\d\-[\d)]*)(\.Arr1-0)(\d).*', bsn)
        ihm, chan, sdm = None, None, None
        if match:
            ihm = match.group(1) + '.wav'
            chan = match.group(3)
            sdm = match.group(1)+match.group(2)+match.group(3) + '.wav'
            return ihm, sdm, chan
        else:
            return None     
         
    entries = []
    with open(file_in) as scps:
        entries = [scp.strip() for scp in scps]

        #first, get headsets only, those will be used for spkids
        ihms = list(filter(lambda x: re.search(r'.*Headset\-\d\-(\d)*\.wav', x), entries))
        ihm_utt2spk = dict(list(map(utt2spk_fun, ihms)))
        ihm2sdms = {k:{} for k in ihm_utt2spk.keys()}
    
        if len(opts.map_ihm2sdm):
            chans = opts.map_ihm2sdm.split(",")
            sdms = list(filter(lambda x: re.search(r'.*Arr.*', x), entries))
            for sdm_path in sdms:
                ihm, sdm, chan = sdm2ihm_and_chan_fun(sdm_path)
                if chan not in chans:
                    print ('Chan {} not in expected chans {}. Skipping'.format(chan, chans))
                    continue
                # pick only distant segments with the corresponding entry in ihm
                if ihm in ihm2sdms: 
                    #print ('Adding {} to {},{}, as {}'.format(sdm, ihm, chan, ihm in ihm2sdms))
                    ihm2sdms[ihm][chan] = sdm
                else:
                    print ('Ihm {} extracted from sdm {} not found in the ihm list'.format(ihm, sdm))
            
            if len(ihm2sdms[ihm].keys()) != len(chans):
                print ('Removed {} utt as the corresponding sdms channels not found'.format(ihm))
                print ('There are two AMI meetings that are missing distant channels')
                ihm2sdms.pop(ihm, None)
                ihm_utt2spk.pop(ihm, None)

        return ihm_utt2spk, ihm2sdms
    return None, None

def mk_ami_path(utt):
    bsn = os.path.basename(utt)
    match = re.match(r'(.*)\.Headset.*', bsn)
    meetid = None
    if match is not None:
        meetid = match.group(1)
    assert (meetid is not None), (
        "Cant extract meeting id from {}. Is this AMI corpus?".format(utt)
    )
    return "{}/audio/{}".format(meetid, bsn)

def main(opts):
    random.seed(opts.seed)

    utt2spk, ihm2sdms = parse_list(opts.train_scp, opts)
    utt2spk_test, ihm2sdms_test = parse_list(opts.test_scp, opts)

    assert utt2spk is not None and ihm2sdms is not None, (
        "Looks like parsing of {} did not suceed".format(opts.train_scp)
    )

    assert utt2spk_test is not None and ihm2sdms_test is not None, (
        "Looks like parsing of {} did not suceed".format(opts.test_scp)
    )

    data_cfg = {'train':{'data':[],
                         'speakers':[]},
                'valid':{'data':[],
                         'speakers':[]},
                'test':{'data':[],
                        'speakers':[]},
                'speakers':[]}

    # get train / valid keys split
    keys = list(utt2spk.keys())
    shuffle(keys)
    N_valid_files = int(len(keys) * opts.val_ratio)
    valid_keys = keys[:N_valid_files]
    train_keys = keys[N_valid_files:]

    train_dur = 0
    for idx, ihm_utt in enumerate(train_keys, start=1):

        print('Processing train file {:7d}/{:7d}'.format(idx, len(train_keys)),
               end='\r')
        
        spk = utt2spk[ihm_utt]
        if spk not in data_cfg['speakers']:
            data_cfg['speakers'].append(spk)
            data_cfg['train']['speakers'].append(spk)

        sdm_utts = ihm2sdms[ihm_utt]
        entry = {'filename':mk_ami_path(ihm_utt), 'spk':spk}
        for chan, sdm_utt in sdm_utts.items():
            entry[chan] = mk_ami_path(sdm_utt)
        data_cfg['train']['data'].append(entry)

        train_dur += get_file_dur(os.path.join(opts.data_root,
                                                   mk_ami_path(ihm_utt)))
    data_cfg['train']['total_wav_dur'] = train_dur
    print()

    valid_dur = 0
    for idx, ihm_utt in enumerate(valid_keys, start=1):

        print('Processing valid file {:7d}/{:7d}'.format(idx, len(valid_keys)),
               end='\r')
        
        spk = utt2spk[ihm_utt]
        if spk not in data_cfg['speakers']:
            data_cfg['speakers'].append(spk)
            data_cfg['valid']['speakers'].append(spk)

        sdm_utts = ihm2sdms[ihm_utt]
        entry = {'filename':mk_ami_path(ihm_utt), 'spk':spk}
        for chan, sdm_utt in sdm_utts.items():
            entry[chan] = mk_ami_path(sdm_utt)
        data_cfg['valid']['data'].append(entry)
        
        valid_dur += get_file_dur(os.path.join(opts.data_root,
                                                   mk_ami_path(ihm_utt)))
    data_cfg['valid']['total_wav_dur'] = valid_dur
    print()

    test_dur = 0
    test_keys = utt2spk_test.keys()
    for idx, ihm_utt in enumerate(test_keys, start=1):

        print('Processing test file {:7d}/{:7d}'.format(idx, len(test_keys)),
               end='\r')
        
        spk = utt2spk_test[ihm_utt]
        if spk not in data_cfg['speakers']:
            data_cfg['speakers'].append(spk)
            data_cfg['test']['speakers'].append(spk)

        sdm_utts = ihm2sdms_test[ihm_utt]
        entry = {'filename':mk_ami_path(ihm_utt), 'spk':spk}
        for chan, sdm_utt in sdm_utts.items():
            entry[chan] = mk_ami_path(sdm_utt)
        data_cfg['test']['data'].append(entry)
        
        test_dur += get_file_dur(os.path.join(opts.data_root,
                                                   mk_ami_path(ihm_utt)))
    data_cfg['test']['total_wav_dur'] = test_dur
    print()

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_ihm2sdm', type=str, default="1,3,5,7",
                        help='Extract VAD segments for these distant channels, on top of close-talk one')
    parser.add_argument('--data_root', type=str, 
                        default='data/LibriSpeech/Librispeech_spkid_sel')
    parser.add_argument('--train_scp', type=str, default=None)
    parser.add_argument('--test_scp', type=str, default=None)
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio to take out of training '
                             'in utterances ratio (Def: 0.1).')
    parser.add_argument('--cfg_file', type=str, default='data/librispeech_data.cfg')
    parser.add_argument('--seed', type=int, default=3)
    
    opts = parser.parse_args()
    main(opts)

