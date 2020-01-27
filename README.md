# Problem Agnostic Speech Encoder (PASE)

This repository contains the official implementations of [PASE](https://arxiv.org/abs/1904.03416) and [PASE+](http://veu.talp.cat/papers/pase_asr_icassp2020.pdf). These are speech waveform encoders trained in a self-supervised manner with the so called worker/minion framework. A PASE model can be used as a speech feature extractor or to pre-train an encoder for our desired end-task, like speech classification such as in ASR, seaker recognition, or emotion recognition, or speech generation such as in voice conversion or [TTS](https://arxiv.org/abs/1906.00733).

![pase+](https://user-images.githubusercontent.com/7583502/72657492-42b88f00-39a5-11ea-9ae6-cf96a1e09042.png)

### NOTE: The old PASE version can be accessed through the tagged commit [`v0.1`](https://github.com/santi-pdp/pase/blob/v0.1_pase/README.md). 

## Requirements

* PyTorch 1.0 or higher
* Torchvision 0.2 or higher
* Install the requirements from `requirements.txt`: `pip install -r requirements.txt`

*NOTE: Edit the cupy-cuda100 requirement in the file if needed depending on your CUDA version. Defaults to 10.0 now*

### Install 

This framework can be installed locally by running:

```
python setup.py install
```

This will allow you to import PASE modules from anywhere.

## Pre-trained Model

The PASE+ parameters used in our most recently [published work](http://veu.talp.cat/papers/pase_asr_icassp2020.pdf) can be found if you [CLICK HERE](https://drive.google.com/open?id=1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW). This ckpt file contains the encoder parameters only, without any worker. This ckpt named `FE_e199.ckpt`, and the configuration file `cfg/frontend/PASE+.cfg` let you build and use the encoder in the following simple manner:

```
from pase.models.frontend import wf_builder
pase = wf_builder('cfg/frontend/PASE+.cfg').eval()
pase.load_pretrained('FE_e199.ckpt', load_last=True, verbose=True)

# Now we can forward waveforms as Torch tensors
import torch
x = torch.randn(1, 1, 100000) # example with random noise to check shape
# y size will be (1, 256, 625), which are 625 frames of 256 dims each
y = pase(x)
```

The encoder can be inserted in any PyTorch model and fine-tuned, just like any
other `nn.Module`.

## Self-Supervised Training Do-It-Yourself (DIY)

### Data preparation

The self-supervised training stage requires the following components to be specified to the training script:

* data root folder: contains `wav` files (or soft links to them) without subfolders.
* trainset statistics file to normalize each worker's output values, computed with the `make_trainset_statistics.py` script.
* dataset configuration `data_cfg` file: contains pointers to train/valid/test splits, among other info.
* front-end (encoder) configuration file: `cfg/frontend/PASE+.cfg`
* workers' configuration file: `cfg/workers/workers+.cfg` 

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

The `make_trainset_statistics.py` script will load a certain amount of training batches with the config file we just generated, and will compute the normalization statistics for the workers to work properly in the self-supervised training. We use this script as follows:

```
python make_trainset_statistics.py --data_root data/LibriSpeech/wavs \
	--data_cfg data/librispeech_data.cfg \
	--net_cfg cfg/workers+.cfg \
	--out_file data/librispeech_stats.pkl 
```

The file `data/librispeech_stats.pkl` will be generated. If this goes too slow, you may try with
a smaller amount of training batches with the `--max_batches 10` argument for example. The default
is 20. Note that the `--net_cfg cfg/workers+.cfg` is supplied so that the script automatically retrieves
the workers that will be active, and the statistics are specific to the workers.

### Training

To train PASE for 150 epochs, with the same hyper-parameters as those in the first published work, execute the following script:

```
python -u train.py --batch_size 32 --epoch 150 --save_path pase_ckpt --num_workers 1 \
	--net_cfg cfg/workers/workers.cfg --fe_cfg cfg/frontend/PASE.cfg \
	--data_cfg data/librispeech_data.cfg --min_lr 0.0005 --fe_lr 0.0005 \
	--data_root data/LibriSpeech/wavs/ --stats data/librispeech_stats.pkl --lrdec_step 30 --lrdecay 0.5
```
Note that `data_root`, `stats` and `data_cfg` are the mentioned data root folder, training statistics file and dataset configuration file (created in previous section).
[TensorboardX](https://github.com/lanpa/tensorboardX) is used during training to dump stats information (stored in `save_path` folder, together with the model checkpoints). The learning rates `min_lr` and `fe_lr` control the worker learning rates and the encoder learning rates respectively. The `lrdec_step` and `lrdecay` params control
the learning rate decay factor and the periodic step at which it is applied, for all components (workers and PASE).

To replicate PASE+ training, execute the following:

```
python -u  train.py --batch_size 16 --epoch 400 --save_path pase+_ckpt \
	       --num_workers 16 --warmup 10000000 --net_cfg cfg/workers/workers+.cfg \
	       --fe_cfg cfg/frontend/PASE+.cfg --do_eval --data_cfg data/librispeech_data_50h.cfg \
	       --min_lr 0.0005 --fe_lr 0.001 --data_root data/LibriSpeech/wavs/ \
	       --dtrans_cfg cfg/distortions/pase+.cfg \
	       --stats data/librispeech_stats.pkl \
	       --chunk_size 32000 \
	       --tensorboard False \
	       --backprop_mode base\
	       --random_scale True\
	       --lr_mode poly
```

Note that the `--lr_mode` allows to choose a different learning rate scheduler. In the `poly` case, a polynomial scheduler updates the LR to reach zero in the end of the programmed epochs. 

The `--dtrans_cfg` flag controls the pointer to the configuration of data augmentation distortions in the form of additive noises, reverberations, etc.

#### Distortions Configuration

The configuration for the distortions allows to control the probability of a distortion being active for a sample in the batch. Hence, distortions are applied on the fly and independently, although with a hard-coded order as programmed in file `pase/transforms.py` (i.e. Reverb happens before Additive, etc.). Note that there are possible distortions:

* Overlap: activated with `overlap_p > 0`
* Additive noise: activated with `noises_p > 0`
* Amplitude clipping: activated with `clip_p > 0`
* Waveform chopping: activated with `chop_p > 0`
* Waveform resampling: activated with `downsample_p > 0`
* Frequency band-drop: activated with `bandrop_p > 0`
* Reverberation: activated with `reverb_p > 0`

Each distortion has a set of parameters that can be controlled, like the impulse response files used to emulate reverberation or pointers to the directories where additive noises are found and the SNRs to be applied randomly. The file `cfg/distortions/pase+.cfg` exemplifies all the possible options to be controlled for the different distortions. A more exhaustive description of each configuration field will be provided soon, as well as pointers to some files that might be used to do a DIY training session with augmentation.

If no `--dtrans_cfg` file is provided, the waveforms are loaded as-is without any change except for a possible random scaling in case `--random_scale True` is supplied in the training command, as shown above.
  

### Running an ASR experiment

In this section, we show how to use PASE+ for a basic speech recognition experiment using the TIMIT dataset (make sure you have it available). The speech recognition experiments reported in the PASE+ paper use standard HMM-DNN technology. The DNN part is composed of the PASE+ encoder coupled with a simple MLP classifier. For the HMM decoding part, we rely on the kaldi toolkit (make sure you have it installed before running the following example).

To run a TIMIT experiment, go to the ASR folder and execute the following command:

```
python run_TIMIT_full_decoding.py $pase_cfg $pase_model $timit_folder $out_folder cfg/MLP_PASE.cfg  cfg/decoder.cfg
```

where $pase_cfg is the path containing the PASE config file (e.g, ../cfg/frontend/PASE+.cfg) and $pase_model contains the path to the PASE weights (e.g,  FE_e199.ckpt).

The script will train the speech recognition system. Once trained the NN, we run the kaldi decoder to retrieve the final sequence of phones. You can take a look into the Phoneme Error Rate by typing:

```
./RESULTS
```

In our case, we achieved a PER=17.2%. Note that natural variations (normally in the order of ± 0.2%) might happen due to different initializations.

### Citation

If using this code, parts of it, or developments from it, please cite our reference:

PASE
```
@inproceedings{Pascual2019,
  author={Santiago Pascual and Mirco Ravanelli and Joan Serrà and Antonio Bonafonte and Yoshua Bengio},
  title={{Learning Problem-Agnostic Speech Representations from Multiple Self-Supervised Tasks}},
  year=2019,
  booktitle={Proc. Interspeech 2019},
  pages={161--165},
  url={http://dx.doi.org/10.21437/Interspeech.2019-2605}
}
```
