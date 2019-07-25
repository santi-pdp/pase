
import soundfile as sf
import sys
import tqdm
import shutil
import os
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import argparse
from timeit import default_timer as timer

def copy_folder(in_folder, out_folder):
    if not os.path.isdir(out_folder):
        print('Replicating dataset structure...')
        beg_t = timer()
        shutil.copytree(in_folder, out_folder, ignore=ig_f)
        end_t = timer()
        print('Replicated structure in {:.1f} s'.format(end_t - beg_t))

def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def handle_multichannel_wav(wav, channel):
    suffixes = {0:'A', 1:'B', 2:'C', 3:'D'}
    if channel > 0:
        assert wav.ndim > 1 and channel < wav.shape[1], (
            "Asked to extract {} channel but file has only sinlge channel".format(channel)
        )
    #wav = wav[:, channel]
    return wav, suffixes[channel]

def segment_signal(args):
    data_root, meetpath, wav_file = args
    wlen = 3200
    wshift = 80
    en_th = 0.3
    smooth_window = 40
    smooth_th_low = 0.25
    smooth_th_high = 0.6
    avoid_sentences_less_that = 24000
    wav_path = os.path.join(data_root, meetpath, wav_file)
    #signal, fs = sf.read(data_folder+wav_file)
    signal, fs = sf.read(wav_path)

    signal, side = handle_multichannel_wav(signal, 0)

    signal = signal / np.max(np.abs(signal))
    
    beg_fr=[0]
    end_fr=[wlen]
    count_fr=0
    en_fr=[]
    
    while end_fr[count_fr] < signal.shape[0]:
        #print(beg_fr[count_fr])
        #print(end_fr[count_fr])
        signal_seg = signal[beg_fr[count_fr]:end_fr[count_fr]]
        en_fr.append(np.mean(np.abs(signal_seg) ** 1))
        beg_fr.append(beg_fr[count_fr]+wshift)
        end_fr.append(beg_fr[count_fr]+wlen+wshift)
        count_fr = count_fr + 1
    
    en_arr=np.asarray(en_fr)
    mean_en=np.mean(en_arr)
    en_bin=(en_arr > mean_en * en_th).astype(int)
    en_bin_smooth=np.zeros(en_bin.shape)
    
    # smooting the window
    for i in range(count_fr):
        if i + smooth_window > count_fr - 1:
            wlen_smooth = count_fr
        else:
            wlen_smooth = i + smooth_window
            
        en_bin_smooth[i] = np.mean(en_bin[i:wlen_smooth])
      
    en_bin_smooth_new = np.zeros(en_bin.shape)
    
    vad = False
    beg_arr_vad=[]
    end_arr_vad=[]
    
    for i in range(count_fr):
        if vad==False:
            
            if en_bin_smooth[i]>smooth_th_high:
                if i<count_fr-1:
                    vad=True
                    en_bin_smooth_new[i]=1
                    beg_arr_vad.append((beg_fr[i])+wlen)
            else:
                en_bin_smooth_new[i]=0                    

        else:
            if i==count_fr-1:
                end_arr_vad.append(end_fr[i]) 
                break
            if en_bin_smooth[i]<smooth_th_low:
                vad=False
                en_bin_smooth_new[i]=0
                end_arr_vad.append((beg_fr[i])+wlen)
            else:
                en_bin_smooth_new[i]=1 
       
    if len(beg_arr_vad) != len(end_arr_vad):
        print('error')
        sys.exit(0)

    # Writing on buffer
    out_buffer = []
    count_seg=0
    for i in range(len(beg_arr_vad)):
        #count_seg_tot=count_seg_tot+1
        if end_arr_vad[i] - beg_arr_vad[i] > avoid_sentences_less_that:
            seg_str = wav_file+ ' ' + str(beg_arr_vad[i]) + ' ' + \
                    str(end_arr_vad[i]) + ' ' + str(count_seg) + '\n'
            out_buffer.append(seg_str)
            count_seg = count_seg + 1
        #else:
        #    count_short = count_short + 1
    return out_buffer

def mk_mic_path(meetid, chan, cond='ihm'):
    assert cond in ['ihm', 'sdm'], (
        "For AMI, cond shoud be in ihm or sdm, got {}".format(cond)
    )
    meetpath = "{}/audio".format(meetid)
    if cond == 'ihm':
        return meetpath, "{}.Headset-{}.wav".format(meetid, chan)
    return meetpath, "{}.Array1-0{}.wav".format(meetid, chan)

def main(opts):
    # copy folder structure
    copy_folder(opts.data_root, opts.out_root)

    headsets = [0, 1, 2, 3] #there is one extra headset in one meeting, but ignore it
    meetings = []
    with open(opts.ami_meeting_ids, 'r') as f:
        for meetid in f:
            meetings.append(meetid.strip()) 

    assert len(meetings) > 0, (
        "Looks like meeting list is empty"
    )

    sdms = []
    if len(opts.map_ihm2sdm) > 0:
        sdms = opts.map_ihm2sdm.split(",")
        for sdm in sdms:
            assert sdm in ['0', '1', '2', '3', '4', '5', '6', '7'], (
                "There are only 8 distant mics in AMI (0...7)"
                "Pick one of them instead {}".format(sdm)
            )

    print ("Preparing AMI for {} meetings,"
            " headset plus {} sdms channels".format(len(meetings), len(sdms)))

    file2spkidx = {}

    for meeting in meetings:
        print ("Processing meeting {}".format(meeting))
        file_out = "{}/{}.Headset.vad".format(opts.out_root, meeting)
        if not os.path.exists(file_out):
            print('VADing signals to build {} list...'.format(file_out))
            with open(file_out, 'w') as f:
                # Paramters for Voice Activity Detection
                wav_lst = []
                for headset in headsets:
                    meetpath, headset_file = mk_mic_path(meeting, headset, 'ihm')
                    wav_lst.append((opts.data_root, meetpath, headset_file))
                
		#commented as it does not work well with aws nfs jsalt stuff
                #pool = mp.Pool(opts.num_workers)
                #for annotations in tqdm.tqdm(pool.imap(segment_signal, wav_lst), 
                #                             total=len(wav_lst)):
		# 
                #    for annotation in annotations:
                #        f.write(annotation)
                for wav_entry in wav_lst:
                   annotations = segment_signal(wav_entry)
                   for annotation in annotations:
                       f.write(annotation)
        else:
            print('[!] Found existing {} file, proceeding with it'.format(file_out))
    
        # now read the list back to create the output chunks
        with open(file_out, 'r') as f:
            fnames = [l.rstrip() for l in f]
            print('Producing segments out of VAD list for ihms...')
            beg_t = timer()
            
            for headset in headsets:
                meetpath, headset_file = mk_mic_path(meeting, headset, 'ihm')
                print ('Working on {}'.format(headset_file))
                signal, fs = sf.read(os.path.join(opts.data_root, meetpath, headset_file))
                signal, side = handle_multichannel_wav(signal, opts.channel)
                signal = signal / np.max(np.abs(signal))
                for li, line in tqdm.tqdm(enumerate(fnames, start=1), total=len(fnames)):
                    wav_file, beg_samp, end_samp, seg_id = line.split(' ')
                    if wav_file != headset_file:
                        # we have joint vad file for all headsets, so its handier for sdms
                        continue
                    segment = signal[int(float(beg_samp)):int(float(end_samp))]
                    out_wav = wav_file.replace('.wav',  '-' + str(seg_id) + '.wav')
                    path_out = os.path.join(opts.out_root, meetpath, out_wav)
                    #print ('\tExporting IHM segment {}'.format(path_out))
                    sf.write(path_out, segment, fs)
                    file2spkidx[out_wav] = wav_file.replace('.wav', '')

            if len(sdms) > 0:
                print('Producing segments out of VAD list for sdms...')
                for sdm in sdms:
                    meetpath, sdm_file = mk_mic_path(meeting, sdm, 'sdm')
                    path_in = os.path.join(opts.data_root, meetpath, sdm_file)

                    if not os.path.exists(path_in):
                        print ('File {} not found. Skipping.'.format(path_in))
                        continue

                    signal, fs = sf.read(path_in)
                    signal = signal / np.max(np.abs(signal))

                    for li, line in tqdm.tqdm(enumerate(fnames, start=1), total=len(fnames)):
                        wav_file, beg_samp, end_samp, seg_id = line.split(' ')
                        segment = signal[int(float(beg_samp)):int(float(end_samp))]
                        wav_file_basename = wav_file.replace('.wav','')
                        wav_out = "{}-{}.Arr1-0{}.wav".format(wav_file_basename, seg_id, sdm)
                        path_out = os.path.join(opts.out_root, meetpath, wav_out)
                        #print ('\tExporting SDM segment {}'.format(path_out))
                        sf.write(path_out, segment, fs)
                        file2spkidx[wav_out] = wav_file_basename
            end_t = timer()

            print('Finalized segments production for meeting : '
                '{}'.format(meeting))
            print('Production time: {:.1f} s'.format(end_t - beg_t))

    np.save(os.path.join(opts.out_root, opts.utt2spk_dict),
            file2spkidx, allow_pickle=True)

    print ('Finished all stuff')
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ami_meeting_ids', type=str, default='ami_split_train.list')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_root', type=str, default=None,
                        help='Directory where files will be stored '
                             '(Def: None).')
    parser.add_argument('--map_ihm2sdm', type=str, default="1,3,5,7",
                        help='Extract VAD segments for these distant channels, on top of close-talk one')
    parser.add_argument('--utt2spk_dict', type=str, default='utt2spk.npy')
    parser.add_argument('--channel', type=int, default=0,
                        help="In case of multi channel file, pick this channel")
    opts = parser.parse_args()
    assert opts.num_workers > 0, opts.num_workers
    if opts.data_root is None:
        raise ValueError('Please specify an input data root (e.g. '
                         'data/LibriSpeech)')
    if opts.out_root is None:
        raise ValueError('Please specify an output data root (e.g. '
                         'data/LibriSpeech_seg)')
    main(opts)

