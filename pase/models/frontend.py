import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from pase.models.WorkerScheduler.encoder import encoder
import torchvision.models as models
try:
    from modules import *
    from aspp import aspp_resblock
    from tdnn import TDNN
except ImportError:
    from .modules import *
    from .gated_cnn import GatedCNN, ResBasicBlock1D
    from .aspp import aspp_resblock
    from .tdnn import TDNN



def wf_builder(cfg_path):
    if cfg_path is not None:
        if isinstance(cfg_path, str):
            with open(cfg_path, 'r') as cfg_f:
                cfg = json.load(cfg_f)
                return wf_builder(cfg)
        elif isinstance(cfg_path, dict):
            if "name" in cfg_path.keys():
                model_name = cfg_path['name']
                if cfg_path['name'] == "asppRes":
                    return aspp_res_encoder(**cfg_path)
                elif model_name == "Resnet50":
                    return Resnet50_encoder(**cfg_path)
                elif model_name == "tdnn":
                    return TDNNFe(**cfg_path)
                elif model_name == "T-MAT":
                    return T_MAT_encoder(**cfg_path)
                else:
                    raise TypeError('Unrecognized frontend type: ', model_name)
            else:
                return WaveFe(**cfg_path)
        else:
            TypeError('Unexpected config for WaveFe')
    else:
        raise ValueError("cfg cannot be None!")


class TDNNFe(Model):
    """ Time-Delayed Neural Network front-end
    """
    def __init__(self, num_inputs=1,
                 sincnet=True,
                 kwidth=641, stride=160,
                 fmaps=128, norm_type='bnorm',
                 pad_mode='reflect',
                 sr=16000, emb_dim=256,
                 activation=None,
                 rnn_pool=False,
                 rnn_layers=1,
                 rnn_dropout=0,
                 rnn_type='qrnn',
                 name='TDNNFe'):
        super().__init__(name=name) 
        # apply sincnet at first layer
        self.sincnet = sincnet
        self.emb_dim = emb_dim
        ninp = num_inputs
        if self.sincnet:
            self.feblock = FeBlock(ninp, fmaps, kwidth, stride,
                                   1, act=activation,
                                   pad_mode=pad_mode,
                                   norm_type=norm_type,
                                   sincnet=True,
                                   sr=sr)
            ninp = fmaps
        # 2 is just a random number because it is not used
        # with unpooled method
        self.tdnn = TDNN(ninp, 2, method='unpooled')
        fmap = self.tdnn.emb_dim
        # last projection
        if rnn_pool:
            self.rnn = build_rnn_block(fmap, emb_dim // 2,
                                       rnn_layers=rnn_layers,
                                       rnn_type=rnn_type,
                                       bidirectional=True,
                                       dropout=rnn_dropout)
            self.W = nn.Conv1d(emb_dim, emb_dim, 1)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.rnn_pool = rnn_pool

    def forward(self, batch, device=None, mode=None):

        # batch possible chunk and contexts, or just forward non-dict tensor
        x, data_fmt = format_frontend_chunk(batch, device)
        if hasattr(self, 'feblock'): 
            h = self.feblock(x)
        
        h = self.tdnn(h)

        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)

        y = self.W(h)

        return format_frontend_output(y, data_fmt, mode)
        """
        if self.training:
            if batched:
                embedding = torch.chunk(y, 3, dim=0)
                chunk = embedding[0]
            else:
                chunk = y
            return embedding, chunk
        else:
            return select_output(h, mode=mode)
        """

class WaveFe(Model):
    """ Convolutional front-end to process waveforms
        into a decimated intermediate representation 
    """
    def __init__(self, num_inputs=1, 
                 sincnet=True,
                 kwidths=[251, 10, 5, 5, 5, 5, 5, 5], 
                 strides=[1, 10, 2, 1, 2, 1, 2, 2], 
                 dilations=[1, 1, 1, 1, 1, 1, 1, 1],
                 fmaps=[64, 64, 128, 128, 256, 256, 512, 512],
                 norm_type='bnorm',
                 pad_mode='reflect', sr=16000,
                 emb_dim=256,
                 rnn_dim=None,
                 activation=None,
                 rnn_pool=False,
                 rnn_layers=1,
                 rnn_dropout=0,
                 rnn_type='qrnn',
                 vq_K=None,
                 vq_beta=0.25,
                 vq_gamma=0.99,
                 norm_out=False,
                 tanh_out=False,
                 resblocks=False,
                 denseskips=False,
                 densemerge='sum',
                 name='WaveFe'):
        super().__init__(name=name) 
        # apply sincnet at first layer
        self.sincnet = sincnet
        self.kwidths = kwidths
        self.strides = strides
        self.fmaps = fmaps
        self.densemerge = densemerge
        if denseskips:
            self.denseskips = nn.ModuleList()
        self.blocks = nn.ModuleList()
        assert len(kwidths) == len(strides)
        assert len(strides) == len(fmaps)
        concat_emb_dim = emb_dim
        ninp = num_inputs
        for n, (kwidth, stride, dilation, fmap) in enumerate(zip(kwidths, 
                                                                 strides,
                                                                 dilations,
                                                                 fmaps), 
                                                             start=1):
            if n > 1:
                # make sure sincnet is deactivated after first layer
                sincnet = False
            if resblocks and not sincnet:
                feblock = FeResBlock(ninp, fmap, kwidth, 
                                     dilation, act=activation,
                                     pad_mode=pad_mode, norm_type=norm_type)
            else:
                feblock = FeBlock(ninp, fmap, kwidth, stride,
                                  dilation,
                                  act=activation,
                                  pad_mode=pad_mode,
                                  norm_type=norm_type,
                                  sincnet=sincnet,
                                  sr=sr)
            self.blocks.append(feblock)
            if denseskips and n < len(kwidths):
                # add projection adapter 
                self.denseskips.append(nn.Conv1d(fmap, emb_dim, 1, bias=False))
                if densemerge == 'concat':
                    concat_emb_dim += emb_dim
            ninp = fmap
        # last projection
        if rnn_pool:
            if rnn_dim is None:
                rnn_dim = emb_dim
            self.rnn = build_rnn_block(fmap, rnn_dim // 2,
                                       rnn_layers=rnn_layers,
                                       rnn_type=rnn_type,
                                       bidirectional=True,
                                       dropout=rnn_dropout)
            self.W = nn.Conv1d(rnn_dim, emb_dim, 1)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.emb_dim = concat_emb_dim
        self.rnn_pool = rnn_pool
        if vq_K is not None and vq_K > 0:
            self.quantizer = VQEMA(vq_K, self.emb_dim,
                                   vq_beta, vq_gamma)
        else:
            self.quantizer = None
        # ouptut vectors are normalized to norm^2 1
        if norm_out:
            if norm_type == 'bnorm':
                self.norm_out = nn.BatchNorm1d(self.emb_dim, affine=False)
            else:
                self.norm_out = nn.InstanceNorm1d(self.emb_dim)
        self.tanh_out = tanh_out

    def fuse_skip(self, input_, skip):
        #print('input_ shape: ', input_.shape)
        #print('skip shape: ', skip.shape)
        dfactor = skip.shape[2] // input_.shape[2]
        if dfactor > 1:
            #print('dfactor: ', dfactor)
            # downsample skips
            # [B, F, T]
            maxlen = input_.shape[2] * dfactor
            skip = skip[:, :, :maxlen]
            bsz, feats, slen = skip.shape
            skip_re = skip.view(bsz, feats, slen // dfactor, dfactor)
            skip = torch.mean(skip_re, dim=3)
            #skip = F.adaptive_avg_pool1d(skip, input_.shape[2])
        if self.densemerge == 'concat':
            return torch.cat((input_, skip), dim=1)
        elif self.densemerge == 'sum':
            return input_ + skip
        else:
            raise TypeError('Unknown densemerge: ', self.densemerge)
        
    def forward(self, batch, device=None, mode=None):
        # batch possible chunk and contexts, or just forward non-dict tensor
        x, data_fmt = format_frontend_chunk(batch, device)
        h = x
        denseskips = hasattr(self, 'denseskips')
        if denseskips:
            dskips = None
            dskips = []
        for n, block in enumerate(self.blocks):
            h = block(h)
            if denseskips and (n + 1) < len(self.blocks):
                # denseskips happen til the last but one layer
                # til the embedding one
                proj = self.denseskips[n]
                dskips.append(proj(h))
                """
                if dskips is None:
                    dskips = proj(h)
                else:
                    h_proj = proj(h)
                    dskips = self.fuse_skip(h_proj, dskips)
                """
        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)
            #y = self.W(h) 
        #else:
        y = self.W(h)
        if denseskips:
            for dskip in dskips:
                # sum all dskips contributions in the embedding
                y = self.fuse_skip(y, dskip)
        if hasattr(self, 'norm_out'):
            y = self.norm_out(y)
        if self.tanh_out:
            y = torch.tanh(y)

        if self.quantizer is not None:
            qloss, y, pp, enc = self.quantizer(y)
            if self.training:
                return qloss, y, pp, enc
            else:
                return y

        return format_frontend_output(y, data_fmt, mode)

class T_MAT_encoder(Model):

  def __init__(self, chunk_size=32000, hidden_dim=512, 
                 kernel_sizes=[11, 11, 11, 11], 
                 strides=[5, 4, 2, 2, 2], fmaps=[64, 64, 128, 512, 1024, 512], 
                 sinc_kernel=251, sinc_stride=1,  
                 dilations=[1, 6, 12, 18], 
                 aspp_fmaps=48, 
                 GCNN_layers=2,
                 GCNN_kernel=1,
                 res_block_count=2,
                 name="T-MAT"):
      super().__init__(name=name)

      if len(strides) % 2 != 1:
        raise ValueError("Number of Conv layers must be odd!!")

      self.sinc = SincConv_fast(1, fmaps[0], sinc_kernel,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=sinc_stride,
                                  pad_mode='reflect'
                                  )

      self.regr_stream = nn.ModuleList()
      for i in range(len(strides)):
        self.regr_stream.append(nn.Sequential(nn.Conv1d(fmaps[i], fmaps[i+1], kernel_sizes[i], strides[i], kernel_sizes[i]//2, bias=False),
                                                     nn.BatchNorm1d(fmaps[i+1]),
                                                     nn.CELU()))


      self.GCNNs = nn.ModuleList()
      self.resblocks = nn.ModuleList()
      compress_factor = sinc_stride
      for i in range(len(strides)):
          compress_factor = compress_factor * strides[i]
          if i % 2 == 0:
              self.GCNNs.append(GatedCNN(seq_len=chunk_size // compress_factor,
                                          n_layers=GCNN_layers,
                                          kernel=GCNN_kernel,
                                          in_chs=fmaps[i+1],
                                          out_chs=fmaps[i+1],
                                          res_block_count=res_block_count,
                                          ans_size=chunk_size // compress_factor))
              
          else:
              self.resblocks.append(ResBasicBlock1D(inplanes=fmaps[i], planes=fmaps[i+2], kwidth=1, stride=strides[i] * strides[i+1],norm_layer=nn.BatchNorm1d))

      # self.cls_stream = zip(self.GCNNs, self.resblocks)
      self.regr_conv = nn.Sequential(nn.Conv1d(hidden_dim, fmaps[-1], 1),
                                    nn.BatchNorm1d(fmaps[-1]),
                                    nn.CELU())
      self.cls_conv = nn.Sequential(nn.Conv1d(fmaps[-1], hidden_dim, 1),
                                    nn.BatchNorm1d(hidden_dim),
                                    nn.CELU())
      self.fusion_module = ASPP(fmaps[-1], hidden_dim, dilations, aspp_fmaps)

      self.emb_dim = hidden_dim

  def forward(self, batch, device=None, mode=None, output_for_representation=False):

      # output_for_representation = True
      # if self.training:
      #     output_for_representation = False

      x, data_fmt = format_frontend_chunk(batch, device)

      sinc_out = self.sinc(x)

      h = sinc_out
      gcnn_out = 0
      res_out = 0
      for i, conv in enumerate(self.regr_stream):
          h = conv(h)

          if i % 2 == 0:
              gcnn_out = self.GCNNs[i // 2](h + res_out)
          
          else:
              res_out = self.resblocks[i // 2](gcnn_out)

      cls_out = self.cls_conv(gcnn_out)
      regr_out = self.fusion_module(h + self.regr_conv(cls_out))

      if not output_for_representation:
          return format_frontend_output(regr_out, data_fmt, mode), format_frontend_output(cls_out, data_fmt, mode)
      else:
          return format_frontend_output(torch.cat((regr_out, cls_out), dim=1), data_fmt, mode)








class aspp_res_encoder(Model):

    def __init__(self, sinc_out, hidden_dim, kernel_sizes=[11, 11, 11, 11], sinc_kernel=251, sinc_stride=1,strides=[10, 4, 2, 2], dilations=[1, 6, 12, 18], fmaps=48, name='aspp_encoder', pool2d=False, rnn_pool=False, rnn_layers=1,rnn_add=False, concat=[False, False, False, True], dense=False):
        super().__init__(name=name)
        self.sinc = SincConv_fast(1, sinc_out, sinc_kernel,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=sinc_stride,
                                  pad_mode='reflect'
                                  )


        self.ASPP_blocks = nn.ModuleList()

        for i in range(len(kernel_sizes)):
            if i == 0:
                self.ASPP_blocks.append(aspp_resblock(sinc_out, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))
            else:
                self.ASPP_blocks.append(aspp_resblock(hidden_dim, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))


        self.rnn_pool = rnn_pool
        self.rnn_add = rnn_add
        self.concat = concat
        assert ((self.rnn_pool and self.rnn_add) or not self.rnn_pool) or self.rnn_pool

        if rnn_pool:
            self.rnn = build_rnn_block(hidden_dim, hidden_dim // 2,
                                       rnn_layers=rnn_layers,
                                       rnn_type='qrnn',
                                       bidirectional=True,
                                       dropout=0)
            self.W = nn.Conv1d(hidden_dim, hidden_dim, 1)


        self.emb_dim = hidden_dim



    def forward(self, batch, device=None, mode=None):

        # batch possible chunk and contexts, or just forward non-dict tensor
        x, data_fmt = format_frontend_chunk(batch, device)

        sinc_out = self.sinc(x)

        out = []
        input = sinc_out
        for i, block in enumerate(self.ASPP_blocks, 0):
            input = block(input)
            if self.concat[i]:
                out.append(input)

        if len(out) > 1:
            out = self.fuse(out)
            out = torch.cat(out, dim=1)
        else:
            out = out[0]



        if self.rnn_pool:
            rnn_out = out.transpose(1, 2).transpose(0, 1)
            rnn_out, _ = self.rnn(rnn_out)
            rnn_out = rnn_out.transpose(0, 1).transpose(1, 2)
            rnn_out = self.W(rnn_out)

        if self.rnn_pool and self.rnn_add:
            h = out + rnn_out
        elif self.rnn_pool and not self.rnn_add:
            h = rnn_out
        else:
            h = out

        return format_frontend_output(h, data_fmt, mode)


    def fuse(self, out):
        last_feature = out[-1]
        for i in range(len(out) - 1):
            out[i] = F.adaptive_avg_pool1d(out[i], last_feature.shape[-1])
        return out

class Resnet50_encoder(Model):

    def __init__(self, sinc_out, hidden_dim, sinc_kernel=251, sinc_stride=1, conv_stride=5, kernel_size=21, pretrained=True,name="Resnet50"):
        super().__init__(name=name)
        self.sinc = SincConv_fast(1, sinc_out, sinc_kernel,
                                  sample_rate=16000,
                                  padding='SAME',
                                  stride=sinc_stride,
                                  pad_mode='reflect'
                                  )

        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=kernel_size, stride=conv_stride, padding= kernel_size // 2, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(64))

        resnet = models.resnet34(pretrained=pretrained)
        self.resnet = nn.Sequential(resnet.layer1,
                                    resnet.layer2,
                                    resnet.layer3,
                                    resnet.layer4
                                    )

        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=[2, 1], stride=1, bias=False))

        self.emb_dim = hidden_dim


    def forward(self, batch, device=None, mode=None):

        # batch possible chunk and contexts, or just forward non-dict tensor
        x, data_fmt = format_frontend_chunk(batch, device)

        sinc_out = self.sinc(x).unsqueeze(1)

        # print(sinc_out.shape)

        conv_out = self.conv1(sinc_out)

        # print(conv_out.shape)

        res_out = self.resnet(conv_out)

        # print(res_out.shape)

        h =self.conv2(res_out).squeeze(2)

        # print(h.shape)

        return format_frontend_output(h, data_fmt, mode)

