
import errno
import sys
import argparse
import logging
import os
import re
import random
import numpy
import soundfile as sf
import multiprocessing
import tqdm
import json
from kaldi_data_dir import KaldiDataDir

#these are global functions, as we use them in Pool workers later so
#must be pickable at any level
def get_wav_and_chan(path):
    #check if raw file, or smth like, in which case we need path and chan
    #sox /export/corpora/CHiME5/audio/train/S03_P09.wav -t wav - remix 1 |
    if re.match(r'.*\|', path):
        #we play with pipe, extract wav and channel
        r = re.search(r'.*\s(.*\.wav)\s.*remix\s([1-9]).*', path)
        if r:
            return r.group(1), int(r.group(2))-1
        return None, None
    return path, None

def process_segment(filein, fileout, beg, end, sigin=None, fs=None):
    #print ("Processing {} to {} for seg ({},{})".format(filein, fileout, beg, end))
    #return True
    if sigin is not None:
        assert fs is not None, (
            "Passed signal as array, but not specified sampling rate (fs)"
        )
    path, chan = get_wav_and_chan(filein)
    if path is None:
        print ("File {} cannot be parsed".format(path))
        return False

    if not os.path.exists(path):
        print ("File {} not found".format(path))
        return False

    if sigin is None:
        sigin, fs = sf.read(path)
    
    beg_i, end_i = int(beg*fs), int(end*fs)
    if beg_i >= end_i or end_i > sigin.shape[0]:
        print ("Cant extract segment {} - {}, as wav {} is {} "\
            .format(beg_i, end_i, filein, sigin.shape[0]))
        return False

    if chan is not None:
        if sigin.ndim > 1 and sigin.shape[1] <= chan:
            print ("File {} has not {} chan present".format(filein, chan))
            return False
        sigout = sigin[beg_i:end_i, chan]
    else:
        sigout = sigin[beg_i:end_i]
    sf.write(fileout, sigout, fs)
    return True

def pool_segment_worker(sess):
    sessid, utts = sess
    tot_success = 0
    for utt in utts:
        uttid, e = utt
        r = process_segment(filein=e['file_in'],
                             fileout=e['file_out'],
                             beg=e['seg_beg'], 
                             end=e['seg_end'])
        if r: tot_success += 1
        #print ("Processed {} from {} success: {}".format(uttid, sessid, r))
    return tot_success

def pool_recording_worker(sess):
    sessid, utts = sess
    tot_success = 0
    sigin, fs, sessfile = None, None, None
    #for each utterance in sessid file (one large file)
    for utt in utts:
        uttid, e = utt
        filein = e['file_in'] 
        fileout = e['file_out']
        beg = e['seg_beg']
        end = e['seg_end']

        #preload the recording once at first request
        if sigin is None:
            sessfile, _ = get_wav_and_chan(filein)
            sigin, fs = sf.read(sessfile)
        else:
            path, _ = get_wav_and_chan(filein)
            assert path == sessfile, (
                "Expected segments share same source wav file {} in session {},"\
                  "but got {} for utt {}".format(sessfile, sessid, path, uttid)
            ) 
        r = process_segment(filein=filein, fileout=fileout, 
                       beg=beg, end=end, 
                       sigin=sigin, fs=fs)
        #r = True
        if r:
            tot_success += 1
            #print ("Proc utt {} (sess {}). From {} to {} into {}".format(uttid, sessid, beg, end, fileout))
        else:
            print ("Failed to process utt {} (sess {})".format(utt, sessid))
    return tot_success

class PasePrep4Chime5(object):
    def __init__(self, out_dir, ihm_dir, sdm_dir=None, num_workers=5):
        assert os.path.exists(out_dir), (
            "Out dir {} expected to exists".format(out_dir)
        )
        self.out_dir = out_dir
        self.name = ihm_dir
        self.ihm = KaldiDataDir(ihm_dir)
        self.sdm = None
        if sdm_dir is not None:
            self.sdm = KaldiDataDir(sdm_dir)
        self.num_workers = num_workers
        self.fs = 16000

    def show_stats(self):
        print ("Stats for {}".format(self.name))
        print ("\t #spkeakers: {}".format(self.ihm.num_spk))
        print ("\t #utterances: {}".format(self.ihm.num_utt))
        print ("\t Tot dur: {} hours".format(self.ihm.total_duarion/3600))

        if self.sdm is not None:
            print ("Stats for {}".format(self.name))
            print ("\t #spkeakers: {}".format(self.sdm.num_spk))
            print ("\t #utterances: {}".format(self.sdm.num_utt))
            print ("\t Tot dur: {} hours".format(self.sdm.total_duarion/3600))

    def get_segments_per_spk(self):
        """Returns 
        """
        for i, spk in enumerate(self.ihm.spk2utt_.keys()):
            utts = self.ihm.spk2utt_[spk]
            print ("Spk {} has {} segments".format(spk, len(utts)))
            for utt in utts:
                seg = self.ihm.utt2segments_[utt].split(" ")
                rec = seg[0]
                wav = self.ihm.utt2wav_[rec]
                print ("Utt {}, rec {}, wav {} ".format(utt, rec, wav))
            if i>10:
                break

    def get_worn_uall_overlap(self):

        def dist2gen(u):
            x = re.sub(r'\.CH[0-9]', '', u)
            x = re.sub(r'\_U0[0-9]\_', '_', x)
            return x
        
        def worn2gen(u):
            x = re.sub(r'\.[L|R]', '', u)
            return x

        utts = self.ihm.utt2spk_.keys()
        utts_worn = list(filter(lambda u: re.search(r'\.[L|R]\-', u), utts))
        utts = self.sdm.utt2spk_.keys()
        utts_dist = list(filter(lambda u: re.search(r'\.CH[0-9]\-', u), utts))
        print ("Found {} worn and {} dist files".format(len(utts_worn), len(utts_dist)))

        gen_utts_worn = dict(list(map(worn2gen, utts_worn)))
        gen_utts_dist = dict(list(map(dist2gen, utts_dist)))
        worn_set = set(gen_utts_worn.keys())
        dist_set = set(gen_utts_dist.keys())

        #gen_utts_worn = list(map(worn2gen, utts_worn))
        #gen_utts_dist = list(map(dist2gen, utts_dist))
        #worn_set = set(gen_utts_worn)
        #dist_set = set(gen_utts_dist)

        isect = worn_set & dist_set

        print (isect)
        print ("Orig lists are {} and {} long".format(len(gen_utts_worn), len(gen_utts_dist)))
        print ("Intersection is {}, while worn and dist are {}, {}".format(len(isect), len(worn_set), len(dist_set)))
        #print (isect)

    def get_Us_for_worn_text(self, min_words_per_seg=2):
        """
        segmentations differ a bit for each of devices vs. worn, thus
        we pair them up based on text hash.
        """

        def mk_txt_id(text, utt):
            ps = utt.split("_")
            return "{}_{} {}".format(ps[0], ps[1], text)

        ihm_utts = list(self.ihm.utt2text_.keys())
        sdm_utts = list(self.sdm.utt2text_.keys())

        random.shuffle(ihm_utts)
        random.shuffle(sdm_utts)

        text2utt_ihm = {}
        skipped_length_ihms = 0
        skipped_doubles = 0
        for utt in ihm_utts:
            txt = self.ihm.utt2text_[utt]
            if len(txt.split(" ")) < min_words_per_seg:
                skipped_length_ihms += 1
                continue
            newid = mk_txt_id(txt, utt)
            if newid in text2utt_ihm:
                #print ("Skipping {} in {} -> {}".format(newid, utt, text2utt_ihm[newid]))
                skipped_doubles += 1
                continue
            text2utt_ihm[newid] = utt

        print ("For IHM, skipped {} too short and {} doubled segs (out of {}). Left {}"\
                   .format(skipped_length_ihms, skipped_doubles, len(ihm_utts), len(text2utt_ihm)))

        text2utt_sdm = {}
        skipped_length_sdms = 0
        skipped_doubles = 0
        for utt in sdm_utts:
            txt = self.sdm.utt2text_[utt]
            if len(txt.split(" ")) < min_words_per_seg:
                skipped_length_sdms += 1
                continue
            newid = mk_txt_id(txt, utt)
            if newid in text2utt_sdm:
                skipped_doubles += 1
                continue
            text2utt_sdm[newid] = utt

        print ("For SDM, skipped {} too short and {} segs (out of {}). Left {}"\
                   .format(skipped_length_sdms, skipped_doubles, len(sdm_utts), len(text2utt_sdm)))

        u1, u2 = set(text2utt_ihm.keys()), set(text2utt_sdm.keys())
        utts_joint = u1 & u2
        print ("Overlap is {}".format(len(utts_joint)))

        spks = self.ihm.spk2utt_.keys()
        spk2chunks = {spk: {'ihm':[], 'sdm':[]} for spk in spks}
        tot = 0
        for idx, utt_joint in enumerate(utts_joint):
            try:
                org_utt_ihm = text2utt_ihm[utt_joint]
                org_utt_sdm = text2utt_sdm[utt_joint]
                #print ("Trying to get {} {}".format(org_utt_ihm, org_utt_sdm))
                spk_ihm = self.ihm.utt2spk_[org_utt_ihm]
                spk_sdm = self.sdm.utt2spk_[org_utt_sdm]
                assert spk_ihm == spk_sdm, (
                    "{} {} not the same".format(spk_ihm, spk_sdm)
                )
                #print ("IHM / SDM are {} {}".format(org_utt_ihm, org_utt_sdm))
                #reco_ihm, beg_ihm, end_ihm = self.ihm.utt2segments_[org_utt_ihm]
                #reco_sdm, beg_sdm, end_sdm = self.sdm.utt2segments_[org_utt_sdm]
                #path_ihm = self.ihm.utt2wav_[reco_ihm]
                #path_sdm = self.sdm.utt2wav_[reco_sdm]
                spk2chunks[spk_ihm]['ihm'].append(org_utt_ihm)
                spk2chunks[spk_ihm]['sdm'].append(org_utt_sdm)
            except KeyError as e:
                print ("Keys {}, {} {} {} {}".format(e, utt_joint, org_utt_ihm, org_utt_sdm, spk_ihm))
                continue


            #print ("Loaded {} {}. {} to {} and {}".format(spk, utt, start, stop, text))
            #print ("Ihm path is {}".format(path))

        return spk2chunks

    def to_data_cfg(self, spk2chunks):

        data_cfg = {'train':{'data':[],
                         'speakers':[],
                         'total_wav_dur':0},
                    'valid':{'data':[],
                         'speakers':[],
                         'total_wav_dur':0},
                    'test':{'data':[],
                        'speakers':[],
                        'total_wav_dur':0},
                    'speakers':[]}

        audio_info = {'ihm':{}, 'sdm':{}}

        valid = 'P42'
        test = 'P41'

        tot_files = 0
        for spk in sorted(spk2chunks.keys()):
            ihm_utts = spk2chunks[spk]['ihm']
            sdm_utts = spk2chunks[spk]['sdm']
            print ("Processing spk {} with {} segs".format(spk, len(ihm_utts)))
            for idx, paths in enumerate(zip(ihm_utts, sdm_utts)):
                #print ("Processing idx {} and paths {}".format(idx, paths))
                org_utt_ihm = paths[0]
                org_utt_sdm = paths[1]

                reco_ihm, beg_ihm, end_ihm = self.ihm.utt2segments_[org_utt_ihm]
                reco_sdm, beg_sdm, end_sdm = self.sdm.utt2segments_[org_utt_sdm]

                path_ihm = self.ihm.utt2wav_[reco_ihm]
                path_sdm = self.sdm.utt2wav_[reco_sdm]

                #out_ihm_file = "{}_{}-{}.wav".format(spk, reco_ihm, idx)
                out_ihm_file = "{}-{}.wav".format(spk, idx)
                out_ihm_path = os.path.join(self.out_dir, out_ihm_file)
                out_sdm_file = "{}_{}-{}.wav".format(spk, reco_sdm, idx)
                #out_sdm_file = "{}_rdm-{}.wav".format(spk, idx)
                out_sdm_path = os.path.join(self.out_dir, out_sdm_file)

                #print ("Proc {} paths {} out_path {}, reco {}".format(idx, paths, out_ihm_file, reco_ihm))

                audio_entry_ihm = {'file_in':path_ihm,
                                   'file_out': out_ihm_path,
                                   'seg_beg': beg_ihm,
                                   'seg_end': end_ihm}
                audio_entry_sdm = {'file_in': path_sdm,
                                   'file_out': out_sdm_path,
                                   'seg_beg': beg_sdm,
                                   'seg_end': end_sdm}

                if reco_ihm not in audio_info['ihm']:
                    audio_info['ihm'][reco_ihm] = []
                if reco_sdm not in audio_info['sdm']:
                    audio_info['sdm'][reco_sdm] = []

                audio_info['ihm'][reco_ihm].append((org_utt_ihm, audio_entry_ihm))
                audio_info['sdm'][reco_sdm].append((org_utt_sdm, audio_entry_sdm))

                dset = 'train'
                if spk in valid:
                    dset = 'valid'
                elif spk in test:
                    dset = 'test'

                entry={'filename': out_ihm_file,
                       '1': out_sdm_file,
                       'spk': spk}
                data_cfg[dset]['data'].append(entry)
                if spk not in data_cfg[dset]['speakers']:
                    data_cfg[dset]['speakers'].append(spk)

                data_cfg[dset]['total_wav_dur'] += int((end_ihm-beg_ihm)*self.fs)

                tot_files += 1

        print ("Data cfg contains {} files in total".format(tot_files))
        print ("Tot training length is {} hours".format(data_cfg['train']['total_wav_dur']/self.fs/3600))

        return data_cfg, audio_info

    def segment_audio(self, audio_info):

        #we want to order by recording, so each one gets loaded only once
        pool = multiprocessing.Pool(processes=self.num_workers)
        sessions = [(k,v) for k,v in audio_info['ihm'].items()]
        print ("Processing IHM sessions....")
        tot_success=0
        for f in tqdm.tqdm(pool.imap(pool_recording_worker, sessions), 
                                         total=len(sessions)):
            tot_success += f
            #print ('F ', f)
        print ("Processed succesfully {} IHM files".format(tot_success))

        sessions = [(k,v) for k,v in audio_info['sdm'].items()]
        print ("Processing SDM sessions....")
        tot_success=0
        for f in tqdm.tqdm(pool.imap(pool_recording_worker, sessions),
                                total=len(sessions)):
            tot_success += f
            #print ('F ', f)
        print ("Processed succesfully {} SDM files".format(tot_success))

if __name__ == "__main__":
    #train_worn_u100k='/mnt/c/work/repos/pase/data_splits/train_worn_u100k'
    #d = PasePrep4Chime5(train_worn_u100k)
    #d.show_stats()
    #d.get_segments_per_spk()
    #d.get_worn_u100k_overlap()

    out_dir='/tmp-corpora/chime5segmented'
    #train_worn='/mnt/c/work/repos/pase/data_splits/train_worn_stereo'
    #train_dist='/mnt/c/work/repos/pase/data_splits/train_uall'
    train_worn='/disks/data1/pawel/repos/kaldi/egs/chime5/s5/data/train_worn_stereo'
    train_dist='/disks/data1/pawel/repos/kaldi/egs/chime5/s5/data/train_uall'
    d = PasePrep4Chime5(out_dir, train_worn, train_dist, num_workers=5)
    d.show_stats()
    #d.get_segments_per_spk()
    spk2chunks = d.get_Us_for_worn_text()
    #if not os.path.exists('spk2chunks.npy'):
    #    spk2chunks = d.get_Us_for_worn_text()
    #    numpy.save('spk2chunks.npy', spk2chunks)
    #else:
    #    spk2chunks = numpy.load('spk2chunks.npy', allow_pickle=True)

    data_cfg, audio_info  = d.to_data_cfg(spk2chunks)

    #sess = audio_info['sdm'].items()
    #for s in sess:
    #    sessid, utts = s
    #    for utt in utts:
    #        uttid, e = utt
    #        filein = e['file_in']
    #        fileout = e['file_out']
    #        print ("Uttid {}, sess {}, fileout {}".format(uttid, sessid, fileout))

    with open("chime5_seg_mathched.cfg", 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))

    d.segment_audio(audio_info)

    #train_u100k='/mnt/c/work/repos/pase/data_splits/train_u100k'
    #d = PasePrep4Chime5(train_u100k)
    #d.show_stats()

    #train_worn='/mnt/c/work/repos/pase/data_splits/train_worn_stereo'
    #d = PasePrep4Chime5(train_worn)
    #d.show_stats()
