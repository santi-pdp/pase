import argparse
import os
import glob
import numpy as np
import librosa
import pandas as pd

def prep_rec(input_rec_path, out_rec_path, sr=16000, out_length_seconds=10):

	try:

		y, s = librosa.load(input_rec_path, sr=sr)

		n_samples = sr*out_length_seconds

		try:
			ridx = np.random.randint(0, len(y)-n_samples)
			librosa.output.write_wav(out_rec_path, y[ridx:(ridx+n_samples)], sr=sr)

			y = y[ridx:(ridx+n_samples)]

		except ValueError:

			mul = int(np.ceil(n_samples/len(y)))
			y = np.tile(y, (mul))[:n_samples]

		librosa.output.write_wav(out_rec_path, y, sr=sr)

		return True

	except:

		return False

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
	args = parser.parse_args()

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

		rec_list = np.random.choice(rec_list, min(args.nrecs, len(rec_list)), replace=False)

		mid_idx=len(rec_list)//3
		train_rec, test_rec = rec_list[mid_idx:], rec_list[:mid_idx]

		for rec in train_rec:
			prep_rec(args.path_to_data+rec, args.out_path+'train/'+lang+'_-_'+rec, sr=args.out_sr, out_length_seconds=args.out_length)
			train_list.append(lang+'_-_'+rec)
			utt2lang[lang+'_-_'+rec]=i

		for rec in test_rec:
			prep_rec(args.path_to_data+rec, args.out_path+'test/'+lang+'_-_'+rec, sr=args.out_sr, out_length_seconds=args.out_length)
			test_list.append(lang+'_-_'+rec)
			utt2lang[lang+'_-_'+rec]=i

	if not os.path.isdir(args.out_path+'lists'):
		os.mkdir(args.out_path+'lists')

	dump_list(train_list, args.out_path+'lists/train_list')
	dump_list(test_list, args.out_path+'lists/test_list')
	np.save(args.out_path+'lists/utt2lang', utt2lang)
