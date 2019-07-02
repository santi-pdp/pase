mkdir /export/corpora/minivoxceleb_100spk
mkdir /export/corpora/minivoxceleb_400spk
mkdir /export/corpora/minivoxceleb_600spk
mkdir /export/corpora/minivoxceleb_800spk


python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_100spk/ --nspk 100
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_600spk/ --nspk 400
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_400spk/ --nspk 600
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_800spk/ --nspk 800

mkdir ../spk_id/minivoxceleb_100spk
mkdir ../spk_id/minivoxceleb_400spk
mkdir ../spk_id/minivoxceleb_600spk
mkdir ../spk_id/minivoxceleb_800spk

mv /export/corpora/minivoxceleb_100spk/lists/* ../spk_id/minivoxceleb_100spk/
mv /export/corpora/minivoxceleb_400spk/lists/* ../spk_id/minivoxceleb_400spk/
mv /export/corpora/minivoxceleb_600spk/lists/* ../spk_id/minivoxceleb_600spk/
mv /export/corpora/minivoxceleb_800spk/lists/* ../spk_id/minivoxceleb_800spk/
