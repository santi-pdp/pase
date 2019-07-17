from .modules import *
from .neural_networks import MLP
import torch

class attention_block(Model):

    def __init__(self, emb_dim, name, options, K, strides, chunksize, mode="concat"):
        super().__init__(name=name)
        self.name = name

        self.emb_dim = emb_dim
        nn_input = self.cal_nn_input_dim(strides, chunksize)
        self.mlp = MLP(options=options, inp_dim= nn_input)
        self.K = K
        self.mode = mode

    def forward(self, hidden, device):

        emb_dim = hidden.shape[1]
        feature_length = hidden.shape[2]
        hidden = hidden.contiguous()
        print(hidden.shape)
        if self.mode == "concat":
            hidden = hidden.view(hidden.shape[0], emb_dim * feature_length)
        if self.mode == "avg_time":
            hidden = hidden.mean(-1)
        if self.mode == "avg_time_batch":
            hidden = hidden.mean(-1).mean(0).unsqueeze(0)
        distribution = self.mlp(hidden)
        hidden = hidden.view(hidden.shape[0], emb_dim, feature_length)

        # distribution = torch.sum(distribution, dim=1)
        _, indices = torch.topk(distribution, dim=1, k=self.K, largest=True, sorted=False)

        # ugly
        # select according to the index
        mask = torch.zeros(hidden.size(), requires_grad=False).to(device).detach()
        for i in range(indices.size()[0]):
            for j in range(indices.size()[1]):
                mask[i, j, :] = 1
        selection = mask * hidden

        return selection, mask

    def cal_nn_input_dim(self, strides, chunk_size):

        if self.mode == "concat":

            compress_factor = 1
            for s in strides:
                compress_factor = compress_factor * s

            if chunk_size % compress_factor != 0:
                raise ValueError('chunk_size should be divisible by the product of the strides factors!')

            nn_input = int(chunk_size // compress_factor) * self.frontend.emb_dim
            print("input_dim of the attention blocks: {}".format(nn_input))
            return nn_input

        if self.mode == "avg_time" or self.mode == "avg_time_batch":

            return self.emb_size




