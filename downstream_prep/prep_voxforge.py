### Copied VAD stuff from https://github.com/idnavid/py_vad_tool

import argparse
import os
import glob
import numpy as np
import librosa
import pandas as pd
import numpy as np
from scipy.io import wavfile

def add_wgn(s,var=1e-4):
	np.random.seed(0)
	noise = np.random.normal(0,var,len(s))
	return s + noise

def enframe(x, win_len, hop_len):

	x = np.squeeze(x)
	if x.ndim != 1:
		raise TypeError("enframe input must be a 1-dimensional array.")
	n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
	x_framed = np.zeros((n_frames, win_len))
	for i in range(n_frames):
		x_framed[i] = x[i * hop_len : i * hop_len + win_len]
	return x_framed

def deframe(x_framed, win_len, hop_len):
	n_frames = len(x_framed)
	n_samples = n_frames*hop_len + win_len
	x_samples = np.zeros((n_samples,1))
	for i in range(n_frames):
		x_samples[i*hop_len : i*hop_len + win_len] = x_framed[i]
	return x_samples

def zero_mean(xframes):
	m = np.mean(xframes,axis=1)
	xframes = xframes - np.tile(m,(xframes.shape[1],1)).T
	return xframes

def compute_nrg(xframes):
	n_frames = xframes.shape[1]
	return np.diagonal(np.dot(xframes,xframes.T))/float(n_frames)

def compute_log_nrg(xframes):
	n_frames = xframes.shape[1]
	raw_nrgs = np.log(compute_nrg(xframes+1e-5))/float(n_frames)
	return (raw_nrgs - np.mean(raw_nrgs))/(np.sqrt(np.var(raw_nrgs)))

def power_spectrum(xframes):
	X = np.fft.fft(xframes,axis=1)
	X = np.abs(X[:,:X.shape[1]/2])**2
	return np.sqrt(X)

def nrg_vad(xframes,percent_thr,nrg_thr=0.,context=5):
	xframes = zero_mean(xframes)
	n_frames = xframes.shape[1]
	
	# Compute per frame energies:
	xnrgs = compute_log_nrg(xframes)
	xvad = np.zeros((n_frames,1))
	for i in range(n_frames):
		start = max(i-context,0)
		end = min(i+context,n_frames-1)
		n_above_thr = np.sum(xnrgs[start:end]>nrg_thr)
		n_total = end-start+1
		xvad[i] = 1.*((float(n_above_thr)/n_total) > percent_thr)
	return xvad

def prep_rec(input_rec_path, out_rec_path, sr=16000, out_length_seconds=10, vad=False):

	try:

		y, s = librosa.load(input_rec_path, sr=sr)
		assert len(y)>s*2

	except:

		print('skipping recording {}'.format(input_rec_path))
		return

	n_samples = sr*out_length_seconds

	if vad:
		win_len = int(s*0.025)
		hop_len = int(s*0.010)
		sframes = enframe(y,win_len,hop_len)
		percent_high_nrg = 0.5
		vad = nrg_vad(sframes,percent_high_nrg)
		vad = deframe(vad,win_len,hop_len)[:len(y)].squeeze()
		y = y[np.where(vad==1)]

	try:
		ridx = np.random.randint(0, len(y)-n_samples)
		librosa.output.write_wav(out_rec_path, y[ridx:(ridx+n_samples)], sr=sr)

		y = y[ridx:(ridx+n_samples)]

	except ValueError:

		try:
			mul = int(np.ceil(n_samples/len(y)))
			y = np.tile(y, (mul))[:n_samples]
		except ZeroDivisionError:
			print('skipping recording {}'.format(input_rec_path))
			return

	librosa.output.write_wav(out_rec_path, y, sr=sr)

def dump_list(list_, path_):

	with open(path_, 'w') as f:
		for el in list_:
			item = el + '\n'
			f.write("%s" % item)

def create_list(data_):

	lang2rec = {}

	for line in data_:

		try:
			lang2rec[line[1]].append(line[0])
		except KeyError:
			lang2rec[line[1]]=[line[0]]
	return lang2rec

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Prep voxforge.')
	parser.add_argument('--path-to-data', type=str, default='./data/')
	parser.add_argument('--path-to-metadata', type=str, default='./data/voxforge.csv')
	parser.add_argument('--out-path', type=str, default='./')
	parser.add_argument('--out-sr', type=int, default=16000)
	parser.add_argument('--out-length', type=int, default=10)
	parser.add_argument('--nrecs', type=int, default=30)
	parser.add_argument('--vad', action='store_true', default=False, help='Enables vad')
	parser.add_argument('--traintest', action='store_true', default=False, help='Enables train test split')
	args = parser.parse_args()

	if args.traintest:

		if not os.path.isdir(args.out_path+'train'):
			os.mkdir(args.out_path+'train')

		if not os.path.isdir(args.out_path+'test'):
			os.mkdir(args.out_path+'test')

	meta_data = pd.read_csv(args.path_to_metadata, sep=',' , header=None).values

	lang2utt = create_list(meta_data)

	train_list, test_list = [], []
	utt2lang = {}

	for i, lang in enumerate(lang2utt):

		print('Language: {}'.format(lang))

		rec_list = lang2utt[lang]

		assert len(rec_list)>1, "Not enough recordings for language {}".format(lang)

		if args.traintest:

			rec_list = np.random.choice(rec_list, min(args.nrecs, len(rec_list)), replace=False)

			mid_idx=len(rec_list)//3
			train_rec, test_rec = rec_list[mid_idx:], rec_list[:mid_idx]

			for rec in train_rec:
				prep_rec(args.path_to_data+rec, args.out_path+'train/'+lang+'_-_'+rec, sr=args.out_sr, out_length_seconds=args.out_length, vad=args.vad)
				train_list.append(lang+'_-_'+rec)
				utt2lang[lang+'_-_'+rec]=i

			for rec in test_rec:
				prep_rec(args.path_to_data+rec, args.out_path+'test/'+lang+'_-_'+rec, sr=args.out_sr, out_length_seconds=args.out_length, vad=args.vad)
				test_list.append(lang+'_-_'+rec)
				utt2lang[lang+'_-_'+rec]=i

		else:

			for rec in rec_list:
				prep_rec(args.path_to_data+rec, args.out_path+lang+'_-_'+rec, sr=args.out_sr, out_length_seconds=args.out_length, vad=args.vad)

	if args.traintest:

		if not os.path.isdir(args.out_path+'lists'):
			os.mkdir(args.out_path+'lists')

		dump_list(train_list, args.out_path+'lists/train_list')
		dump_list(test_list, args.out_path+'lists/test_list')
		np.save(args.out_path+'lists/utt2lang', utt2lang)
