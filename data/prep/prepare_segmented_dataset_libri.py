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

def segment_signal(args):
    data_root, wav_file = args
    wlen = 3200
    wshift = 80
    en_th = 0.3
    smooth_window = 40
    smooth_th_low = 0.25
    smooth_th_high = 0.6
    avoid_sentences_less_that = 24000
    wav_path = os.path.join(data_root, wav_file)
    #signal, fs = sf.read(data_folder+wav_file)
    signal, fs = sf.read(wav_path)
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
            seg_str = wav_file + ' ' + str(beg_arr_vad[i]) + ' ' + \
                    str(end_arr_vad[i]) + ' ' + str(count_seg) + '\n'
            out_buffer.append(seg_str)
            count_seg = count_seg + 1
        #else:
        #    count_short = count_short + 1
    return out_buffer

def main(opts):
    data_folder = opts.data_root
    file_lst = opts.file_list
    file_out = opts.file_out
    save_path = opts.out_root

    # copy folder structure
    copy_folder(opts.data_root, opts.out_root)
    
    if not os.path.exists(file_out):
        print('VADing signals to build {} list...'.format(file_out))
        pool = mp.Pool(opts.num_workers)

        with open(file_out, 'w') as f:
            # Paramters for Voice Activity Detection

            # Readline all the files
            with open(file_lst, 'r') as lst_f:
                wav_lst = [(data_folder, line.rstrip()) for line in lst_f]

                count=1
                count_seg_tot=0
                count_short=0

                wi = 1
                for annotations in tqdm.tqdm(pool.imap(segment_signal, wav_lst), 
                                             total=len(wav_lst)):
                    for annotation in annotations:
                        f.write(annotation)
    else:
        print('[!] Found existing {} file, proceeding with it'.format(file_out))
            
    # now read the list back to create the output chunks
    with open(file_out, 'r') as f:
        fnames = [l.rstrip() for l in f]
        print('Producing segments out of VAD list...')
        beg_t = timer()
        for li, line in tqdm.tqdm(enumerate(fnames, start=1), total=len(fnames)):
            wav_file, beg_samp, end_samp, seg_id = line.split(' ')
            signal, fs = sf.read(os.path.join(opts.data_root, wav_file))
            signal = signal / np.max(np.abs(signal))
            signal = signal[int(float(beg_samp)):int(float(end_samp))]
            path_out = os.path.join(opts.out_root, wav_file)
            path_out = path_out.replace('.flac', '-' + str(seg_id) + '.wav')
            sf.write(path_out, signal, fs)
        end_t = timer()
        print('Finalized segments production to output path: '
              '{}'.format(opts.out_root))
        print('Production time: {:.1f} s'.format(end_t - beg_t))
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--file_list', type=str, default='data/libri_all_tr.lst')
    parser.add_argument('--file_out', type=str, default='data/libri_snt_vad.lst')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--out_root', type=str, default=None,
                        help='Directory where files will be stored '
                             '(Def: None).')
    opts = parser.parse_args()
    assert opts.num_workers > 0, opts.num_workers
    if opts.data_root is None:
        raise ValueError('Please specify an input data root (e.g. '
                         'data/LibriSpeech)')
    if opts.out_root is None:
        raise ValueError('Please specify an output data root (e.g. '
                         'data/LibriSpeech_seg)')
    main(opts)

