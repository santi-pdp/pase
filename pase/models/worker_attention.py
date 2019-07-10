from .modules import *
from neural_networks import MLP



class attention_block(Model):

    def __init__(self, emb_dim, name, options):
        super.__init__(self)
        self.name = name
        options['dnn_lay'] = str(emb_dim) + str(emb_dim) * (len(options['dnn_drop']) - 1)
        self.mlp = MLP(options)

    def forward(self, hidden):
        return self.mlp(hidden)
