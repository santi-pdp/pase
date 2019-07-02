import argparse
import os
import glob
import numpy as np
import librosa

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

def clean_dir(dir_path):

	file_list = glob.glob(dir_path+'/*.*')

	if len(file_list)>0:
		for file_ in file_list:
			os.remove(file_)

def dump_list(list_, path_):

	with open(path_, 'w') as f:
		for el in list_:
			item = el + '\n'
			f.write("%s" % item)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Prep vox.')
	parser.add_argument('--path-to-data', type=str, default='./data/')
	parser.add_argument('--out-path', type=str, default='./')
	parser.add_argument('--out-sr', type=int, default=16000)
	parser.add_argument('--out-length', type=int, default=10)
	parser.add_argument('--nspk', type=int, default=100)
	parser.add_argument('--ntrials', type=int, default=10)
	args = parser.parse_args()

	if not os.path.isdir(args.out_path+'train'):
		os.mkdir(args.out_path+'train')

	if not os.path.isdir(args.out_path+'test'):
		os.mkdir(args.out_path+'test')

	spk_list = np.random.choice(os.listdir(args.path_to_data), args.nspk, replace=False)

	train_list, test_list = [], []
	utt2spk = {}

	for i, spk in enumerate(spk_list):

		print('Speaker: {}'.format(spk))

		folder_list = os.listdir(args.path_to_data + spk + '/')
		rec_list = []

		for folder in folder_list:
			folder_recs = os.listdir(args.path_to_data + spk + '/' + folder + '/')

			for rec in folder_recs:
				rec_list.append(args.path_to_data + spk + '/' + folder + '/' + rec)

		train_rec, test_rec = np.random.choice(rec_list, 2, replace=False)

		trials=0
		success=False
		while not success and trials<args.ntrials:

			clean_dir(args.path_to_data+'train')
			clean_dir(args.path_to_data+'test')

			train_rec, test_rec = np.random.choice(rec_list, 2, replace=False)

			train_utt = train_rec.split('/')[-1]
			train_folder = train_rec.split('/')[-2]
			test_utt = test_rec.split('/')[-1]
			test_folder = test_rec.split('/')[-2]

			success = prep_rec(train_rec, args.out_path+'train/'+spk+'_-_'+train_folder+'_-_'+train_utt, sr=args.out_sr, out_length_seconds=args.out_length) and prep_rec(test_rec, args.out_path+'test/'+spk+'_-_'+test_folder+'_-_'+test_utt, sr=args.out_sr, out_length_seconds=args.out_length)

			trials+=1

		if trials>=args.ntrials:
			print('Failed!!')
			exit(1)

		train_list.append(spk+'_-_'+train_folder+'_-_'+train_utt)
		test_list.append(spk+'_-_'+test_folder+'_-_'+test_utt)
		utt2spk[train_list[-1]] = i
		utt2spk[test_list[-1]] = i

	if not os.path.isdir(args.out_path+'lists'):
		os.mkdir(args.out_path+'lists')

	print('Any overlap between train and test lists: {}'.format(bool(set(train_list) & set(test_list))))

	dump_list(train_list, args.out_path+'lists/train_list')
	dump_list(test_list, args.out_path+'lists/test_list')
	np.save(args.out_path+'lists/utt2spk', utt2spk)
