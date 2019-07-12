from .modules import *
from .neural_networks import MLP
import torch

class attention_block(Model):

    def __init__(self, emb_dim, name, options, K):
        super().__init__(name=name)
        self.name = name
        # options['dnn_lay'] = str(emb_dim) + "," + str(emb_dim)

        self.mlp = MLP(options=options, inp_dim=emb_dim)
        self.K = K

    def forward(self, hidden, device):
        distribution = self.mlp(hidden)
        distribution = torch.sum(distribution, dim=2)
        _, indices = torch.topk(distribution, dim=1, k=self.K, largest=True, sorted=False)

        # ugly
        # select according to the index
        mask = torch.zeros(hidden.size(), requires_grad=False).to(device).detach()
        for i in range(indices.size()[0]):
            for j in range(indices.size()[1]):
                mask[i, j, :] = 1
        selection = mask * hidden

        return selection, mask



