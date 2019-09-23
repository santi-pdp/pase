from .Minions.minions import *
from .Minions.cls_minions import *
from .attention_block import attention_block
from .frontend import wf_builder
from .WorkerScheduler.encoder import *
import numpy as np
import torch

class pase_attention(Model):

    def __init__(self,
                 frontend=None,
                 frontend_cfg=None,
                 att_cfg=None,
                 minions_cfg=None,
                 cls_lst=["mi", "cmi", "spc"],
                 regr_lst=["chunk", "lps", "mfcc", "prosody"],
                 adv_lst=[],
                 K=40,
                 att_mode="concat",
                 avg_factor=0,
                 chunk_size=16000,
                 pretrained_ckpt=None,
                 name="adversarial"):

        super().__init__(name=name)
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions'
                             ' config with at least 1 minion. '
                             'GIMME SOMETHING TO DO.')

        # init frontend
        print(frontend_cfg)
        self.frontend = wf_builder(frontend_cfg)

        # init all workers
        # putting them into 3 lists
        self.cls_lst = cls_lst
        self.reg_lst = regr_lst
        self.adv_lst = adv_lst

        ninp = self.frontend.emb_dim
        self.regression_workers = nn.ModuleList()
        self.classification_workers = nn.ModuleList()
        self.adversarial_workers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        # nn_input = self.cal_nn_input_dim(frontend_cfg['strides'], chunk_size)

        # auto infer the output dim of first nn layer
        att_cfg['dnn_lay'] += "," + str(ninp)

        for type, cfg_lst in minions_cfg.items():
            for cfg in cfg_lst:

                if type == 'cls':
                    cfg['num_inputs'] = ninp
                    self.classification_workers.append(cls_worker_maker(cfg, ninp))
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor,att_mode))

                elif type == 'regr':
                    cfg['num_inputs'] = ninp
                    minion = minion_maker(cfg)
                    self.regression_workers.append(minion)
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor,att_mode))

                elif type == 'adv':
                    cfg['num_inputs'] = ninp
                    minion = minion_maker(cfg)
                    self.adversarial_workers.append(minion)
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor,att_mode))

                else:
                    raise TypeError('Unrecognized worker type: ', type)

        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt, load_last=True)

    def forward(self, x, alpha=1, device=None):

        # forward the encoder
        # x[chunk, context, rand] => y[chunk, context, rand], chunk

        h, chunk = self.frontend(x, device)

        # forward all attention blocks
        # chunk => new_chunk, indices
        new_hidden = {}
        for att_block in self.attention_blocks:
            hidden, indices = att_block(chunk, device)
            new_hidden[att_block.name] = (hidden, indices)

        # forward all classification workers
        # h => chunk

        preds = {}
        labels = {}
        for worker in self.regression_workers:
            hidden, _ = new_hidden[worker.name]
            y = worker(hidden, alpha)
            preds[worker.name] = y
            labels[worker.name] = x[worker.name].to(device).detach()
            if worker.name == 'chunk':
                labels[worker.name] = x['cchunk'].to(device).detach()

        # forward all regression workers
        # h => y, label

        for worker in self.classification_workers:
            hidden, mask = new_hidden[worker.name]
            h = [hidden, h[1] * mask, h[2] * mask]
            if worker.name == "spc":
                y, label = worker(hidden, alpha, device)
            elif worker.name == "overlap":
                y = worker(hidden, alpha)
                label = x[worker.name].to(device).detach()
            else:
                y, label = worker(h, alpha, device=device)
            preds[worker.name] = y
            labels[worker.name] = label

        return h, chunk, preds, labels


class pase(Model):

    def __init__(self,
                 frontend=None,
                 frontend_cfg=None,
                 minions_cfg=None,
                 cls_lst=["mi", "cmi", "spc"],
                 regr_lst=["chunk", "lps", "mfcc", "prosody"],
                 pretrained_ckpt=None,
                 name="adversarial"):
        super().__init__(name=name)
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions'
                             ' config with at least 1 minion. '
                             'GIMME SOMETHING TO DO.')

        # init frontend
        print("pase config ==>", frontend_cfg)
        self.frontend = wf_builder(frontend_cfg)
        self.frontend_name = frontend_cfg["name"]


        # init all workers
        # putting them into two lists
        self.cls_lst = cls_lst
        self.reg_lst = regr_lst

        ninp = self.frontend.emb_dim
        self.regression_workers = nn.ModuleList()
        self.classification_workers = nn.ModuleList()
        # these are unparameterized
        self.regularizer_workers = []
        self.fwd_cchunk = False

        count_cat = 0
        if "concat" in frontend_cfg.keys():
            for cat in frontend_cfg['concat']:
                if cat:
                    count_cat += 1
        if count_cat == 0:
            count_cat = 1

        ninp *= count_cat

        print("==>concat features from {} levels".format(count_cat))
        print("==>input size for workers: {}".format(ninp))

        for type, cfg_lst in minions_cfg.items():

            for cfg in cfg_lst:

                if type == 'cls':
                    cfg['num_inputs'] = ninp
                    self.classification_workers.append(cls_worker_maker(cfg, ninp))

                elif type == 'regr':
                    cfg['num_inputs'] = ninp
                    minion = minion_maker(cfg)
                    self.regression_workers.append(minion)
                
                elif type == 'regu':
                    if 'cchunk' in cfg['name']:
                        # cchunk will be necessary
                        self.fwd_cchunk = True
                    minion = minion_maker(cfg)
                    self.regularizer_workers.append(minion)

        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt, load_last=True)

    def forward(self, x, alpha=1, device=None):

        # forward the encoder
        # x[chunk, context, rand, cchunk] => y[chunk, context, rand, cchunk], chunk
        x_ = dict((k, v) for k, v in x.items())
        if not self.fwd_cchunk:
            # remove key if it exists
            x_.pop('cchunk', None)
        h = self.frontend(x_, device)
        T_MAT = (self.frontend_name == "T-MAT")
        if T_MAT:
            T_MAT = True
            regr_h, cls_h = h
            regr_hidden, regr_chunk = regr_h
            cls_hidden, cls_chunk = cls_h

        if len(h) > 1:
            assert len(h) == 2, len(h)
            h, chunk = h

        # forward all classification workers
        # h => chunk
        preds = {}
        labels = {}

        for worker in self.regularizer_workers:
            preds[worker.name] = chunk
            # select forwarded data from the PASE frontend according to 
            # from last position which must be 'cchunk'
            # This way PASE(chunk) is enforced to fall over PASE(cchunk)
            labels[worker.name] = h[-1].to(device).detach()

        if T_MAT:
            chunk = regr_chunk
        for worker in self.regression_workers:
            y = worker(chunk, alpha)
            preds[worker.name] = y
            labels[worker.name] = x[worker.name].to(device).detach()
            if worker.name == 'chunk':
                labels[worker.name] = x[worker.name].to(device).detach()

        # forward all regression workers
        # h => y, label
        if T_MAT:
            chunk = cls_chunk
            h = cls_hidden
        for worker in self.classification_workers:
            if worker.name == "spc" or worker.name == "gap":
                y, label = worker(chunk, alpha, device=device)
            elif worker.name == "overlap":
                y = worker(chunk, alpha)
                label = x[worker.name].to(device).detach()
            else:
                y, label = worker(h, alpha, device=device)
            preds[worker.name] = y
            labels[worker.name] = label

        return h, chunk, preds, labels
