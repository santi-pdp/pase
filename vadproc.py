from scipy.io import wavfile
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import webrtcvad
import struct
import argparse


def main(opts):
    rate, wav = wavfile.read(opts.in_file)
    vad = webrtcvad.Vad()
    vad.set_mode(opts.vad_mode)
    win_len = int(opts.win_len * rate + .5)
    # y is to plot (wav time resolution), vads contains one point per frame
    y = []
    vads = []
    frames = []
    for beg_i in np.arange(0, len(wav), win_len):
        end_i = min(beg_i + win_len, len(wav))
        frame = wav[beg_i:end_i]
        frames.append(frame)
        if len(frame) < win_len:
            P = win_len - len(frame)
            frame = np.concatenate((frame,
                                    np.zeros((P,), dtype=frame.dtype)),
                                    axis=0)
        xb = struct.pack("%dh" % len(frame), *frame)
        is_speech = vad.is_speech(xb,
                                  sample_rate = rate)
        y_i = 1 if is_speech else 0
        vads.append(y_i)
        y += [y_i] * win_len

    if opts.show:
        plt.subplot(2, 1, 1)
        plt.plot(wav)
        plt.subplot(2, 1, 2)
        plt.plot(y)
        plt.show()

    if opts.trim_sil > 0:
        assert opts.out_file is not None 
        # post-process to find silence regions over
        # this value, and delete those pieces of speech
        count0 = 0
        fcandidates = []
        out_samples = []
        max_sil = int(np.ceil(((opts.trim_sil / 1000) * 16000) / win_len))
        frame_len = len(frames[0])
        for idx, (y_i, frame) in enumerate(zip(vads, frames), start=1):
            if y_i == 0:
                count0 += 1
                fcandidates.extend(frame.tolist())
            if y_i == 1 or idx >= len(frames):
                # change detected, process all previous counts
                # and frame candidates, to know whether to 
                # discard all but max allowed or to leave them alone
                if count0 >= max_sil:
                    # discard all candidates but last trim_sil ones
                    K = max_sil * frame_len
                    fcandidates = fcandidates[-K:]
                if len(fcandidates) > 0:
                    out_samples += fcandidates[:]
                    fcandidates = []
                frame = frame.tolist()
                out_samples += frame[:]
                count0 = 0
        out_samples = np.array(out_samples, dtype=np.int16)
        R = 100 - ((len(out_samples) / len(wav)) * 100.)
        if opts.verbose:
            print('Input num samples: ', len(wav))
            print('Output num samples: ', len(out_samples))
            print('Discarded {} % of samples'.format(R))
        if opts.show:
            plt.subplot(3,1,1)
            plt.plot(wav)
            plt.subplot(3,1,2)
            plt.plot(y)
            plt.subplot(3,1,3)
            plt.plot(out_samples)
            plt.show()
        sf.write(opts.out_file, out_samples, rate, 'PCM_16')
        if opts.out_log is not None:
            with open(opts.out_log, 'w') as out_log_f:
                out_log_f.write('isamples\tosamples\treduction[%]\n{}\t{}\t{:.2f}\n'
                                ''.format(len(wav), len(out_samples), R))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str, default=None,
                        help='Path to wav file to be processed (Def: None).')
    parser.add_argument('--trim_sil', type=float,
                        default=0,
                        help='Silence regions over this value '
                             'will be trimmed. The value 0 has no effect. '
                             'Specify in milisec, please (Def: 0).')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--out_file', type=str, default=None)
    parser.add_argument('--out_log', type=str, default=None,
                        help='Will write log in this file if specified')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--win_len', type=float, default=0.02, 
                        help='In seconds (Def: 0.02).')
    parser.add_argument('--vad_mode', type=int, default=3)


    opts = parser.parse_args()
    main(opts)
