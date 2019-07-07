### from: https://github.com/PiSchool/spoken-language-id/blob/master/voxforge.py


import os
import re
import csv
import tarfile
import argparse
from collections import Counter
from collections import OrderedDict

import wget
import requests


def make_args():
	parser = argparse.ArgumentParser(description="VoxForge dataset downloader.")
	parser.add_argument('--per-user', default=1, type=int, help="Limit the number of recordings per user")
	parser.add_argument('--per-user-archives', default=1, type=int, help="Limit the number of archives per user")
	parser.add_argument('-d', '--output-dir', default='voxforge_samples', help="Directory to output wave files to")
	parser.add_argument('-l', '--output-log', default='voxforge_samples.csv', help="Metadata about downloaded files")
	return parser.parse_args()


if __name__ == '__main__':
	base_url = 'http://www.repository.voxforge1.org/downloads/{lang}/Trunk/Audio/Original/48kHz_16bit/{archive}'
	languages = {
		'Italian': 'it',
		'French': 'fr',
		'Portuguese': 'pt',
		'German': 'de',
		'English': 'SpeechCorpus',
		'Spanish': 'es',
		'Persian': 'fa',
		'Russian': 'ru',
		'Turkish': 'tr',
		'Chinese': 'zh',
	}

	args = make_args()
	os.makedirs(args.output_dir, exist_ok=True)

	log_file = open(args.output_log, 'w')
	log_csv = csv.writer(log_file, lineterminator='\n')
	for lang_name, lang_code in languages.items():
		download_url = base_url.format(lang=lang_code, archive='')

		# Download an HTML page with archive names
		print("Downloading archives for {}.".format(lang_name))
		resp = requests.get(download_url)

		# Generate a list of (archive_name, user_name) pairs
		archives = re.findall(r'((\w+)-[\w-]+\.tgz)', resp.text)

		# Remove duplicates, retain order
		archives = OrderedDict.fromkeys(archives)

		# Pick a single archive from each user
		user_archives = Counter()
		user_recordings = Counter()
		for archive, user in archives:
			if user_archives[user] >= args.per_user_archives:
				# We have enough archives of this user
				continue

			user_archives[user] += 1

			# Download an archive
			download_url = base_url.format(lang=lang_code, archive=archive)

			# Remove the downloaded archive
			archive_file = '/tmp/foo.tgz'
			try:
				os.remove(archive_file)
			except OSError:
				pass

			# Download the archive
			wget.download(download_url, out=archive_file)
			print()

			per_archive_count = 0
			with tarfile.open(archive_file, errorlevel=2) as tar:
				for member in tar.getmembers():
					# Find a member that is a wave file, e.g. archive-name/wav/it-0123.wav
					wav_filename = re.match(r'([\w-]+)/.+/([\w-]+\.wav)', member.name)
					if wav_filename is not None:
						if user_recordings[user] >= args.per_user:
							# We have enough files from this user
							break

						user_recordings[user] += 1
						per_archive_count += 1
						archive_name = wav_filename.group(1)

						# Rename conveniently
						member.name = '{archive}-{wav}'.format(archive=archive_name, wav=wav_filename.group(2))

						# Skip if exists
						if os.path.isfile(os.path.join(args.output_dir, member.name)):
							continue

						tar.extract(member, path=args.output_dir)
						log_csv.writerow([member.name, lang_name, user, user_recordings[user]])
						log_file.flush()
			print("Extracted {} files from {}".format(per_archive_count, archive))

		print("Recordings by {} users.".format(len(user_archives)))
	log_file.close()
