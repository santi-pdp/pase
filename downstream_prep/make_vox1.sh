mkdir /export/corpora/minivoxceleb_40spk
mkdir /export/corpora/minivoxceleb_100spk
mkdir /export/corpora/minivoxceleb_400spk
mkdir /export/corpora/minivoxceleb_600spk
mkdir /export/corpora/minivoxceleb_800spk

python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_40spk/ --nspk 40
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_100spk/ --nspk 100
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_400spk/ --nspk 400
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_600spk/ --nspk 600
python prep_voxceleb.py --path-to-data /export/corpora/voxceleb1/train_wav/ --out-path /export/corpora/minivoxceleb_800spk/ --nspk 800

mkdir ../spk_id/minivoxceleb_40spk
mkdir ../spk_id/minivoxceleb_100spk
mkdir ../spk_id/minivoxceleb_400spk
mkdir ../spk_id/minivoxceleb_600spk
mkdir ../spk_id/minivoxceleb_800spk

mv /export/corpora/minivoxceleb_40spk/lists/* ../spk_id/minivoxceleb_40spk/
mv /export/corpora/minivoxceleb_100spk/lists/* ../spk_id/minivoxceleb_100spk/
mv /export/corpora/minivoxceleb_400spk/lists/* ../spk_id/minivoxceleb_400spk/
mv /export/corpora/minivoxceleb_600spk/lists/* ../spk_id/minivoxceleb_600spk/
mv /export/corpora/minivoxceleb_800spk/lists/* ../spk_id/minivoxceleb_800spk/

mv ../spk_id/minivoxceleb_40spk/train_list ../spk_id/minivoxceleb_40spk/minivox_tr_list.txt
mv ../spk_id/minivoxceleb_100spk/train_list ../spk_id/minivoxceleb_100spk/minivox_tr_list.txt
mv ../spk_id/minivoxceleb_400spk/train_list ../spk_id/minivoxceleb_400spk/minivox_tr_list.txt
mv ../spk_id/minivoxceleb_600spk/train_list ../spk_id/minivoxceleb_600spk/minivox_tr_list.txt
mv ../spk_id/minivoxceleb_800spk/train_list ../spk_id/minivoxceleb_800spk/minivox_tr_list.txt

mv ../spk_id/minivoxceleb_40spk/test_list ../spk_id/minivoxceleb_40spk/minivox_test_list.txt
mv ../spk_id/minivoxceleb_100spk/test_list ../spk_id/minivoxceleb_100spk/minivox_test_list.txt
mv ../spk_id/minivoxceleb_400spk/test_list ../spk_id/minivoxceleb_400spk/minivox_test_list.txt
mv ../spk_id/minivoxceleb_600spk/test_list ../spk_id/minivoxceleb_600spk/minivox_test_list.txt
mv ../spk_id/minivoxceleb_800spk/test_list ../spk_id/minivoxceleb_800spk/minivox_test_list.txt
