"""
    2016 Pawel Swietojanski
"""
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

logger = logging.getLogger(__name__)


def make_directory(directory):
    """Creates a directory on disk, first checks whether it exists"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise IOError('Could not create a direcotry' + directory)

def execute_shell_cmd():
    pass


class KaldiDataDir(object):
    """Reads/writes Kaldi data directories"""

    def __init__(self, directory, preload=True):

        self.directory = directory

        self.uttids_ = {}
        self.spkids_ = {}

        self.utt2spk_ = {}
        self.spk2utt_ = {}
        self.utt2gender_ = {}
        self.utt2dur_ = {}
        self.utt2text_ = {}
        self.utt2segments_ = {}
        self.utt2text_ = {}
        self.utt2wav_ = {}
        self.utt2native_ = {}
        self.reco2file_and_channel_ = {}

        self.utt2reco_ = {}
        # below two fields are not-typical for Kaldi data organisation, but it
        # using our metadata to generate scoring labels for different
        # conditions data was collected in (room, noise-on/noise-off, etc.)
        # utt2cond_ maps an utterance to stm compatible condition ids
        # (space separated)
        self.utt2cond_ = {}
        # cond2desc_ contains details on conditions, for example:
        # "<ID>" "<COL_HD>" "<DESC>"
        # "M" "Male" "Male Talkers" (see also to_stm method)
        # "N1" "Noise" "Noise on"
        # "N0" "Noise" "Noise off"
        self.cond2desc_ = {}

        if preload:
            self.read_datadir()

    @property
    def num_spk(self):
        return len(self.spk2utt_)
    
    @property
    def num_utt(self):
        return len(self.utt2spk_)
    
    @property
    def total_duarion(self):
        tot_dur = 0
        for d in self.utt2dur_.values():
            tot_dur += d
        return tot_dur

    def load(self):
        """This function should implement exact backend mapper (CSV, SQL, etc.)
        """
        raise NotImplementedError()

    def ids_constructor(self, row):
        """
        Shall implement by specific recipes on how to compose required utt ids
        given some unstructured data (for example, when fetching raw data from
        database/s3)
        """
        raise NotImplementedError()

    def write_datadir(self):
        """Writes internal state to Kaldi compatible directory"""
        try:
            self.__write_utt2spk()
            self.__write_spk2utt()
            self.__write_utt2gender()
            self.__write_utt2cond()
            self.__write_cond2desc()
            self.__write_utt2dur()
            self.__write_segments()
            self.__write_text()
            self.__write_wavscp()
            self.__write_reco2file_and_channel()
        except:
            raise Exception('Duming files failed')

    def read_datadir(self):
        """Writes internal state to Kaldi compatible directory"""
        
        a = self.__read_utt2spk()
        b = self.__read_wavscp()
        c = self.__read_text()
        d = self.__read_segments()

        l = [a,b,c,d]
        assert any(l) is not False, (
            "Expected files utt2spk, wavscp and text exists {}".format(l)
        )

        if not self.__read_spk2utt():
            print ("Read utt 2 spk skipped")
        if not self.__read_utt2gender():
            print ("Read utt 2 gender skipped")
        if not self.__read_utt2cond():
            print ("Read utt 2 cond skipped")
        if not self.__read_cond2desc():
            print ("Read cond 2 desc skipped")
        if not self.__read_utt2dur():
            print ("Read utt2dur skipped")
        if not self.__read_reco2file_and_channel():
            print ("Reading rec2file_and_channel skipped")

    def to_stm(self):
        """
        Maps internal representation into stm that shall be used for scoring.
        This will allows us to group different types of data by metadata
        (and thus get a separate WER result for each, on top of aggregate one)

        See for details:
            http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/infmts.htm

        The format follows like this (excerpt of above desc):

        STM :== <F> <C> <S> <BT> <ET> [ <LABEL> ] transcript . . .

        For example:
        ;; LABEL "<ID>" "<COL_HD>" "<DESC>"
        ;; LABEL "M" "Male" "Male Talkers"
        ;; LABEL "F" "Female" "Female Talkers"
        ;; LABEL "01" "Story 1" "Business news"
        ;; LABEL "00" "Not in Story" "Words or Phrases not contained
                       in a story"
        940328 1 A 4.00 18.10 <O,F,00> FROM LOS ANGELES
        940328 1 B 18.10 25.55 <O,M,01> MEXICO IN TURMOIL
        """
        with open(self.directory + "/stm", 'w') as stm:
            stm.write(";; LABEL \"M\" \"Male\" \"Male Talkers\"\n")
            stm.write(";; LABEL \"F\" \"Female\" \"Female Talkers\"\n")
            stm.write(";; LABEL \"L1\" \"Natives\" \"Native speakers\"\n")
            stm.write(";; LABEL \"L0\" \"Non-natives\" "
                      "\"Non native speakers\"\n")
            for key in sorted(self.cond2desc_.keys()):
                stm.write(";; LABEL " + self.cond2desc_[key] + "\n")

            for key in sorted(self.uttids_.keys()):
                wav = os.path.basename(self.utt2wav_[key])
                stm.write(wav + " A " + self.utt2spk_[key] + " 0 30.0 <O," +
                          self.utt2gender_[key] + "," +
                          self.utt2native_[key] + "," +
                          self.utt2cond_[key] + "> " +
                          self.utt2text_[key] + "\n")

    def __check_consistency(self):
        pass

    def __write_dict(self, fname, wdict):
        if wdict is None or len(wdict.keys()) < 1:
            return
        with open(os.path.join(self.directory, fname), 'w') as fh:
            for key, val in sorted(wdict.items()):
                fh.write(key + " " + val + "\n")

    def __read_dict(self, fname, wdict):
        try:
            with open(os.path.join(self.directory, fname), 'r') as fh:
                for idx, line in enumerate(fh):
                    line = line.strip()
                    try:
                        key, val = re.split(' ', line, maxsplit=1)
                        if key in wdict:
                            logger.warning("Warning, " + key + " existsted and was overriden")
                        wdict[key] = val.strip()
                    except ValueError as ve:
                        print ("Incorrect line no. {} of {} ({}).".format(idx, fname, line))
                        print (" Error is \"{}\"".format(ve))
            return True
        except:
            return False

    def __write_utt2spk(self):
        self.__write_dict("utt2spk", self.utt2spk_)

    def __read_utt2spk(self):
        return self.__read_dict("utt2spk", self.utt2spk_)

    def __write_spk2utt(self):
        self.__write_dict("spk2utt", self.spk2utt_)

    def __read_spk2utt(self):
        spk2utt = {}
        r = self.__read_dict("spk2utt", spk2utt)
        if r:
            self.spk2utt_ = {k: v.split(" ") for k,v in spk2utt.items()}
        return r

    def __write_utt2gender(self):
        self.__write_dict("utt2gender", self.utt2gender_)

    def __read_utt2gender(self):
        return self.__read_dict("utt2gender", self.utt2gender_)

    def __write_utt2cond(self):
        self.__write_dict("utt2cond", self.utt2cond_)

    def __read_utt2cond(self):
        return self.__read_dict("utt2cond", self.utt2cond_)

    def __write_cond2desc(self):
        self.__write_dict("cond2desc", self.cond2desc_)

    def __read_cond2desc(self):
        return self.__read_dict("cond2desc", self.cond2desc_)

    def __write_utt2dur(self):
        self.__write_dict("utt2dur", self.utt2dur_)

    def __read_utt2dur(self):
        utt2dur = {}
        r = self.__read_dict("utt2dur", utt2dur)
        if r:
            self.utt2dur_ = {k: float(v) for k,v in utt2dur.items()}
        return r

    def __write_segments(self):
        self.__write_dict("segments", self.utt2segments_)

    def __read_segments(self):
        def seg2list(s):
            l = s.split(" ")
            assert len(l) == 3, (
                "Incorrect seg format"
            )
            l[1] = float(l[1])
            l[2] = float(l[2])
            return l

        utt2seg = {}
        r = self.__read_dict("segments", utt2seg)
        if r:
            self.utt2segments_ = {k: seg2list(v) for k,v in utt2seg.items()}
        return r

    def __write_text(self):
        self.__write_dict("text", self.utt2text_)

    def __read_text(self):
        return self.__read_dict("text", self.utt2text_)

    def __write_utt2native(self):
        self.__write_dict("utt2native", self.utt2native_)

    def __read_utt2native(self):
        return self.__read_dict("utt2native", self.utt2native_)

    def __write_wavscp(self):
        self.__write_dict("wav.scp", self.utt2wav_)

    def __read_wavscp(self):
       return  self.__read_dict("wav.scp", self.utt2wav_)

    def __write_reco2file_and_channel(self):
        self.__write_dict("reco2file_and_channel", self.reco2file_and_channel_)

    def __read_reco2file_and_channel(self):
        return self.__read_dict("rec2file_and_channel", self.reco2file_and_channel_)


def kaldi_env(kaldi_root):
    kaldi_root = kaldi_root.strip()
    os.environ['KALDI_ROOT'] = kaldi_root
    os.environ['PATH'] = os.popen(
        'echo $KALDI_ROOT/src/bin:'
        '$KALDI_ROOT/tools/openfst/bin:'
        '$KALDI_ROOT/src/fstbin/:'
        '$KALDI_ROOT/src/gmmbin/:'
        '$KALDI_ROOT/src/featbin/:'
        '$KALDI_ROOT/src/lm/:'
        '$KALDI_ROOT/src/sgmmbin/:'
        '$KALDI_ROOT/src/sgmm2bin/:'
        '$KALDI_ROOT/src/fgmmbin/:'
        '$KALDI_ROOT/src/latbin/:'
        '$KALDI_ROOT/src/nnetbin:'
        '$KALDI_ROOT/src/nnet2bin:'
        '$KALDI_ROOT/src/nnet3bin:'
        '$KALDI_ROOT/src/online2bin/:'
        '$KALDI_ROOT/src/ivectorbin/:'
        '$KALDI_ROOT/src/lmbin/').readline().strip() + ':' + os.environ['PATH']



         



