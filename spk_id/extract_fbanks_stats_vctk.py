import os
import glob
import pickle
import numpy as np
from ahoproc_tools.io import read_aco_file


TR_GUIA="../data/VCTK/spk_id/vctk_tr.scp"
FB_STATS="stats/vctk_fbanks.stats"
DATA_ROOT="../data/VCTK/spk_id/fbanks"
ORDER=40

with open(TR_GUIA) as tr_guia_f:
    tr_files = [l.rstrip() for l in tr_guia_f]

X = []
for ii, tr_file in enumerate(tr_files, start=1):
    print('Reading mfcc {}/{}...'.format(ii, len(tr_files)))
    bname = os.path.splitext(tr_file)[0]
    tr_fpath = os.path.join(DATA_ROOT, bname + '.fb.npy')
    cc = np.load(tr_fpath)
    X.append(cc)

X = np.concatenate(X, axis=0)
with open(FB_STATS, 'wb') as stats_f:
    x_m = list(np.mean(X, axis=0))
    x_std = list(np.std(X, axis=0))
    pickle.dump({'mean':x_m, 'std':x_std}, stats_f)
