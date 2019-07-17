from .modules import *
from .neural_networks import MLP
import torch
import torch.nn.functional as F

class attention_block(Model):

    def __init__(self, emb_dim, name, options, K, strides, chunksize, avg_factor=0, mode="concat"):
        super().__init__(name=name)
        
        self.name = name
        self.mode = mode
        self.emb_dim = emb_dim
        self.avg_factor = avg_factor
        nn_input = self.cal_nn_input_dim(strides, chunksize)

        self.mlp = MLP(options=options, inp_dim= nn_input)
        self.K = K
        self.avg_init=True
    

    def forward(self, hidden, device):
        batch_size = hidden.shape[0]
        feature_length = hidden.shape[2]
        hidden = hidden.contiguous()
    
        if self.mode == "concat":
            hidden_att = hidden.view(hidden.shape[0], self.emb_dim * feature_length)
        if self.mode == "avg_time":
            hidden_att = hidden.mean(-1)
        if self.mode == "avg_time_batch":
            hidden_att = hidden.mean(-1).mean(0).unsqueeze(0)

        distribution = self.mlp(hidden_att)

        if self.avg_init:
            self.running_dist = self.init_running_avg(batch_size).to(device).detach()
            self.avg_init = False

        self.running_dist = self.running_dist.detach() * self.avg_factor + distribution * (1 - self.avg_factor)
        

        distribution = self.running_dist
        
        # distribution = torch.sum(distribution, dim=1)
        _, indices = torch.topk(distribution, dim=1, k=self.K, largest=True, sorted=False)
        
        # select according to the index
        mask = torch.zeros(distribution.size(), requires_grad=False).to(device).detach()
        mask = mask.scatter(1,indices,1).unsqueeze(-1).repeat(1,1,feature_length)

        selection = mask * hidden

        return selection, mask

    def cal_nn_input_dim(self, strides, chunk_size):

        if self.mode == "concat":

            compress_factor = 1
            for s in strides:
                compress_factor = compress_factor * s

            if chunk_size % compress_factor != 0:
                raise ValueError('chunk_size should be divisible by the product of the strides factors!')

            nn_input = int(chunk_size // compress_factor) * self.emb_dim
            print("input_dim of the attention blocks: {}".format(nn_input))
            return nn_input

        if self.mode == "avg_time" or self.mode == "avg_time_batch":

            return self.emb_dim


    def init_running_avg(self, batch_size):
        dist = torch.randn(self.emb_dim).float()
        dist = dist.unsqueeze(0).repeat(batch_size,1)
        dist = F.softmax(dist,dim=1)
        return dist






