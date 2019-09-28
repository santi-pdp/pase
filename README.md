# Problem Agnostic Speech Encoder (PASE)

This repository is the official implementation of [PASE](https://arxiv.org/abs/1904.03416), a speech waveform encoder trained in a self-supervised framework with the so called workers. PASE can be used as a speech feature extractor or can be used to pre-train a network that perform a speech classification task such as speech recognition, speaker identification, emotion classification, etc.

![SimplePASE](https://user-images.githubusercontent.com/7583502/55691535-5ad39a80-599f-11e9-81aa-4ea5ad949d09.png)

## Requirements

* PyTorch 1.0 or higher
* Torchvision 0.2 or higher
* Install the deps in the requirements file: `pip install -r requirements.txt`

*IMPORTANTLY: change the cupy package cuda version depending on your system. It deafults to CUDA 10.0 with cupy-cuda100*

## Pre-trained Model

The PASE parameters used in the published work can be found <a href='http://veu.talp.cat/models/PASE.ckpt'>here</a>. This `ckpt` file
contains the encoder parameters only, without any worker. This, and the configuration file `cfg/PASE.cfg` let you build and use the 
encoder in the following simple manner:

```
from pase.models.frontend import wf_builder
pase = wf_builder('cfg/PASE.cfg')
pase.eval()
pase.load_pretrained('PASE.ckpt', load_last=True, verbose=True)

# Now we can forward waveforms as Torch tensors
import torch
x = torch.randn(1, 1, 100000)
# y size will be (1, 100, 625), which are 625 frames of 100 dims each
y = pase(x)
```

The encoder can be inserted in any PyTorch model and fine-tuned, just like any
other `nn.Module`.

## Self-Supervised Training Do-It-Yourself (DIY)

### Data preparation

The self-supervised training stage requires the following components to be specified to the training script:

* data root folder: contains `wav` files (or soft links to them) without subfolders.
* trainset statistics file to normalize each worker's output values
* dataset configuration `data_cfg` file: contains pointers to train/valid/test splits, among other info.
* front-end (encoder) configuration file: `cfg/PASE.cfg`
* workers' configuration file: `cfg/workers.cfg` 

#### Making the dataset config file

To make the dataset configuration file the following files have to be provided:

* training files list `train_scp`: contains a `wav` file name per line (without directory names), including `.wav` extension.
* test files list `test_scp`: contains a `wav` file name per line (without directory names), including `.wav` extension.
* dictionary with `wav` filename -> integer speaker class (speaker id) correspondence (same filenames as in train/test lists).

An example of each of these files can be found in the `data/` folder of the repo. Build them based on your data files.

_NOTE: The `filename2spkclass` dictionary is required to create a train/valid/test split which holds out some speakers from training, such that
self-supervised training validation tracks the workers' losses with unseen identities (thus to truly generalize). Those labels,
however, are not used during training for this is an unsupervised framework._

We use the following script to create our dataset configuration file (`--cfg_file`):

```
python unsupervised_data_cfg_librispeech.py --data_root data/LibriSpeech/wavs \
	--train_scp data/LibriSpeech/libri_tr.scp --test_scp data/LibriSpeech/libri_te.scp \
	--libri_dict data/LibriSpeech/libri_dict.npy --cfg_file data/librispeech_data.cfg

```

#### Making the trainset statistics file

The `make_trainset_statistics.py` script will load a certain amount of training batches with the config file we just generated,
and will compute the normalization statistics for the workers to work properly in the self-supervised training. We use this script
as follows:

```
python make_trainset_statistics.py --data_root data/LibriSpeech/wavs \
	--data_cfg data/librispeech_data.cfg \
	--out_file data/librispeech_stats.pkl
```

The file `data/librispeech_stats.pkl` will be generated. If this goes too slow, you may try with
a smaller amount of training batches with the `--max_batches 10` argument for example. The default
is 20.

**Now we have the ingredients to train our PASE model.**

### Training

To train PASE for 150 epochs, with the same hyper-parameters as those in the published work, execute the following script:

```
python -u train.py --batch_size 32 --epoch 100 --save_path pase_ckpt --num_workers 1 \
	--net_cfg cfg/workers.cfg --fe_cfg cfg/PASE.cfg \
	--do_eval --data_cfg data/librispeech_data.cfg --min_lr 0.0005 --fe_lr 0.0005 \
	--data_root data/LibriSpeech/wavs/ --stats data/librispeech_stats.pkl --lrdec_step 30 --lrdecay 0.5
```

Note that `data_root`, `stats` and `data_cfg` are the mentioned data root folder, training statistics file and dataset configuration file (created in previous section).
[TensorboardX](https://github.com/lanpa/tensorboardX) is used during training to dump stats information (stored in `save_path` folder, together with the model checkpoints). The `do_eval` flag activates validation 
tracking which will be printed out to tensorboard. The learning rates `min_lr` and `fe_lr` control the worker learning rates and the encoder learning rates respectively. The `lrdec_step` and `lrdecay` params control
the learning rate decay factor and the periodic step at which it is applied, for all components (workers and PASE). 

_NOTE: there is an additional possible worker that can be activated during training, the adversarial worker. This works as a regularizer to shape up the latent space of PASE like a Normal distribution, 
basically following the_ [adversarial autoencoder](https://arxiv.org/abs/1511.05644) _methodology. The `warmup` parameter can be specified as an argument, which is huge by default. This is because
we do not activate it for these experiments, but it can be used if it is of any interest by the reader/user._


## Authors

* Santiago Pascual (Universitat Politècnica de Catalunya - Barcelona)
* Mirco Ravanelli (MILA, Université de Montréal - Montréal)
* Joan Serrà (Telefónica Research - Barcelona)
* Antonio Bonafonte (Universitat Politècnica de Catalunya - Barcelona)
* Yoshua Bengio (MILA, Université de Montréal - Montréal, CIFAR Fellow)
