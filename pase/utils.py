import json
import shlex
import subprocess
import random
import torch
import torch.nn as nn
try:
    from .losses import *
except ImportError:
    from losses import *
import random
from random import shuffle
from pase.models.discriminator import *
import torch.optim as optim
from torch.autograd import Function


def pase_parser(cfg_fname, batch_acum=1, device='cpu', do_losses=True,
                frontend=None):
    with open(cfg_fname, 'r') as cfg_f:
        cfg_all = json.load(cfg_f)
        if do_losses:
            # change loss section
            for i, cfg in enumerate(cfg_all):
                loss_name = cfg_all[i]['loss']
                if hasattr(nn, loss_name):
                    # retrieve potential r frames parameter
                    r_frames = cfg_all[i].get('r', None)
                    # loss in nn Modules
                    cfg_all[i]['loss'] = ContextualizedLoss(getattr(nn, loss_name)(),
                                                            r=r_frames)
                else:
                    if loss_name == 'LSGAN' or loss_name == 'GAN':
                        dnet_cfg = {}
                        if 'DNet_cfg' in cfg_all[i]:
                            dnet_cfg = cfg_all[i].pop('DNet_cfg')
                        dnet_cfg['frontend'] = frontend
                        # make DNet
                        DNet =  RNNDiscriminator(**dnet_cfg)
                        if 'Dopt_cfg' in cfg_all[i]:
                            Dopt_cfg = cfg_all[i].pop('Dopt_cfg')
                            Dopt = optim.Adam(DNet.parameters(),
                                                 Dopt_cfg['lr'])
                        else:
                            Dopt = optim.Adam(DNet.parameters(), 0.0005)
                    Dloss = 'L2' if loss_name == 'LSGAN' else 'BCE'
                    cfg_all[i]['loss'] = WaveAdversarialLoss(DNet, Dopt,
                                                             loss=Dloss,
                                                             batch_acum=batch_acum,
                                                             device=device)
        return cfg_all

def worker_parser(cfg_fname, batch_acum=1, device='cpu', do_losses=True,
                frontend=None):
    with open(cfg_fname, 'r') as cfg_f:
        cfg_list = json.load(cfg_f)
        if do_losses:
            # change loss section
            for type, cfg_all in cfg_list.items():

                for i, cfg in enumerate(cfg_all):
                    loss_name = cfg_all[i]['loss']
                    if hasattr(nn, loss_name):
                        # retrieve potential r frames parameter
                        r_frames = cfg_all[i].get('r', None)
                        # loss in nn Modules
                        cfg_all[i]['loss'] = ContextualizedLoss(getattr(nn, loss_name)(),
                                                                r=r_frames)
                    else:
                        if loss_name == 'LSGAN' or loss_name == 'GAN':
                            dnet_cfg = {}
                            if 'DNet_cfg' in cfg_all[i]:
                                dnet_cfg = cfg_all[i].pop('DNet_cfg')
                            dnet_cfg['frontend'] = frontend
                            # make DNet
                            DNet =  RNNDiscriminator(**dnet_cfg)
                            if 'Dopt_cfg' in cfg_all[i]:
                                Dopt_cfg = cfg_all[i].pop('Dopt_cfg')
                                Dopt = optim.Adam(DNet.parameters(),
                                                     Dopt_cfg['lr'])
                            else:
                                Dopt = optim.Adam(DNet.parameters(), 0.0005)
                        Dloss = 'L2' if loss_name == 'LSGAN' else 'BCE'
                        cfg_all[i]['loss'] = WaveAdversarialLoss(DNet, Dopt,
                                                                 loss=Dloss,
                                                                 batch_acum=batch_acum,
                                                                 device=device)
            cfg_list[type] = cfg_all
        print(cfg_list)
        return cfg_list


def build_optimizer(opt_cfg, params):
    if isinstance(opt_cfg, str):
        with open(opt_cfg, 'r') as cfg_f:
            opt_cfg = json.load(cfg_f)
    opt_name = opt_cfg.pop('name')
    if 'sched' in opt_cfg:
        sched_cfg = opt_cfg.pop('sched')
    else:
        sched_cfg = None
    opt_cfg['params'] = params
    opt = getattr(optim, opt_name)(**opt_cfg)
    if sched_cfg is not None:
        sname = sched_cfg.pop('name')
        sched_cfg['optimizer'] = opt
        sched = getattr(optim.lr_scheduler, sname)(**sched_cfg)
        return opt, sched
    else:
        return opt, None

def chunk_batch_seq(X, seq_range=[90, 1000]):
    bsz, nfeats, slen = X.size()
    min_seq = seq_range[0]
    max_seq = min(slen, seq_range[1])
    # sample a random chunk size
    chsz = random.choice(list(range(min_seq, max_seq)))
    idxs = list(range(slen - chsz))
    beg_i = random.choice(idxs)
    return X[:, :, beg_i:beg_i + chsz]

def kfold_data(data_list, utt2class, folds=10, valid_p=0.1):
    # returns the K lists of lists, so each k-th component
    # is composed of 3 sub-lists
    #idxs = list(range(len(data_list)))
    # shuffle the idxs first
    #shuffle(idxs)
    # group by class first
    classes = set(utt2class.values())
    items = dict((k, []) for k in classes)
    for data_el in data_list:
        items[utt2class[data_el]].append(data_el)
    lens = {}
    test_splits = {}
    for k in items.keys():
        shuffle(items[k])
        lens[k] = len(items[k])
        TEST_SPLIT_K = int((1. / folds) * lens[k])
        test_splits[k] = TEST_SPLIT_K
    lists = []
    beg_i = dict((k, 0) for k in test_splits.keys())
    # now slide a window per fold
    for fi in range(folds):
        test_split = []
        train_split = []
        valid_split = []
        print('-' * 30)
        print('Fold {} splits:'.format(fi))
        for k, data in items.items():
            te_split = data[beg_i[k]:beg_i[k] + test_splits[k]]
            test_split += te_split
            tr_split = data[:beg_i[k]] + data[beg_i[k] + test_splits[k]:]
            # select train and valid splits
            tr_split = tr_split[int(valid_p * len(tr_split)):]
            va_split = tr_split[:int(valid_p * len(tr_split))]
            train_split += tr_split
            valid_split += va_split
            print('Split {} train: {}, valid: {}, test: {}'
                  ''.format(k, len(tr_split), len(va_split), len(te_split)))
        # build valid split within train_split
        lists.append([train_split, valid_split, test_split])
    return lists

class AuxiliarSuperviser(object):

    def __init__(self, cmd_file, save_path='.'):
        self.cmd_file = cmd_file
        with open(cmd_file, 'r') as cmd_f:
            self.cmd = [l.rstrip() for l in cmd_f]
        self.save_path = save_path

    def __call__(self, iteration, ckpt_path, cfg_path):
        assert isinstance(iteration, int)
        assert isinstance(ckpt_path, str)
        assert isinstance(cfg_path, str)
        for cmd in self.cmd:
            sub_cmd = cmd.replace('$model', ckpt_path)
            sub_cmd = sub_cmd.replace('$iteration', str(iteration))
            sub_cmd = sub_cmd.replace('$cfg', cfg_path)
            sub_cmd = sub_cmd.replace('$save_path', self.save_path)
            print('Executing async command: ', sub_cmd)
            #shsub = shlex.split(sub_cmd)
            #print(shsub)
            p = subprocess.Popen(sub_cmd,
                                shell=True)


def get_grad_norms(model, keys=[]):
    grads = {}
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        accept = False
        for key in keys:
            # match substring in collection of model keys
            if key in k:
                accept = True
                break
        if not accept:
            continue
        if param.grad is None:
            print('WARNING getting grads: {} param grad is None'.format(k))
            continue
        grads[k] = torch.norm(param.grad).cpu().item()
    return grads

def sample_probable(p):
    return random.random() < p

def zerospeech(shape, eps=1e-14):
    S = np.random.randn(shape) * eps
    return S.astype(np.float32)


class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha

        return output, None

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=65536,
                                  log_scale_min=None, reduce=True):
    """ https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    y_hat = y_hat.permute(0,2,1)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x

