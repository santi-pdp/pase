# Problem Agnostic Speech Encoder (PASE)

This repository contains the official implementations of [PASE](https://arxiv.org/abs/1904.03416) and [PASE+](http://veu.talp.cat/papers/pase_asr_icassp2020.pdf). These are speech waveform encoders trained in a self-supervised manner with the so called worker/minion framework. A PASE model can be used as a speech feature extractor or to pre-train an encoder for our desired end-task, like speech classification such as in ASR, seaker recognition, or emotion recognition, or speech generation such as in voice conversion or [TTS](https://arxiv.org/abs/1906.00733).

![pase+](https://user-images.githubusercontent.com/7583502/72657492-42b88f00-39a5-11ea-9ae6-cf96a1e09042.png)

### NOTE: The old PASE version can be accessed through the tagged commit `v0.1`. 

## Requirements

* PyTorch 1.0 or higher
* Torchvision 0.2 or higher
* Install the requirements from `requirements.txt`: `pip install -r requirements.txt`

*NOTE: Edit the cupy-cuda100 requirement in the file if needed depending on your CUDA version. Defaults to 10.0 now*

## Pre-trained Model

The PASE+ parameters used in our most recently [published work](http://veu.talp.cat/papers/pase_asr_icassp2020.pdf) can be found if you [CLICK HERE](https://drive.google.com/open?id=1xwlZMGnEt9bGKCVcqDeNrruLFQW5zUEW). This ckpt file contains the encoder parameters only, without any worker. This, and the configuration file cfg/PASE+.cfg let you build and use the encoder in the following simple manner:

```
from pase.models.frontend import wf_builder
pase = wf_builder('cfg/PASE+.cfg').eval()
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

**TO BE UPDATED Soon with latest PASE+ commands**

#### Making the dataset config file

**TO BE UPDATED Soon with latest PASE+ commands**

#### Making the trainset statistics file

**TO BE UPDATED Soon with latest PASE+ commands**

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
