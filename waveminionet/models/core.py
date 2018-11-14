from modules import *
from frontend import *
from minions import *


class Waveminionet(Model):

    def __init__(self, frontend=None, frontend_cfg=None,
                 minions_cfg=None,
                 name='Waveminionet'):
        super().__init__(name=name)
        # augmented wav processing net
        # it trains simultaneously with many tasks
        # forcing a hierarchy of abstraction to distill
        # the contents within waveforms
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions'
                             ' config with at least 1 minion. '
                             'GIMME SOMETHING TO DO.')
        if frontend is not None:
            self.frontend = frontend
        else:
            if frontend_cfg is None:
                # default params
                self.frontend = WaveFe()
            else:
                self.frontend = WaveFe(**frontend_cfg)
        # -------- MINION STACK --------
        self.minions = nn.ModuleList()
        ninp = self.frontend.emb_dim
        for minion_cfg in minions_cfg:
            minion_cfg['num_inputs'] = ninp
            minion = minion_maker(minion_cfg)
            self.minions.append(minion)
            if minion.skip:
                nouts = minion.hidden_size
                # acumulate num of inputs (concat skip connection)
                ninp += nouts


    def forward(self, x):
        fe_h = self.frontend(x)
        print('front-end inference: ', fe_h.size())
        h = fe_h
        outs = {}
        for mi, minion in enumerate(self.minions, start=1):
            y, h_ = minion(h)
            print('minion {}: {} -> {}'.format(mi, h.size(), h_.size()))
            print('{}-> {}'.format(' ' * 36, y.size()))
            if minion.skip:
                h_c = torch.cat((h, h_), dim=1)
                print('Concat skip {} = [{}, {}]'.format(h_c.size(), 
                                                         h.size(),
                                                         h_.size()))
                h = h_c
            else:
                h = h
                print('Skip flow is cut here!')
            print('=' * 70)
            outs[minion.name] = y
        return outs, h

if __name__ == '__main__':
    import json
    wmodel = Waveminionet(
                          minions_cfg=[
                              {'num_outputs':1,
                               'dropout':0.2,
                               'name':'recon',
                               'type':'decoder',
                              },
                              {'num_outputs':257,
                               'dropout':0.2,
                               'name':'lps'
                              },
                              {'num_outputs':40,
                               'dropout':0.2,
                               'name':'mfcc'
                              },
                              {'num_outputs':4,
                               'dropout':0.2,
                               'name':'prosody'
                              },
                              {'num_outputs':1,
                               'dropout':0.2,
                               'name':'mi'
                              },
                          ]
                         )
    print(wmodel)
    x = torch.randn(1, 1, 8000)
    outs, y = wmodel(x)
    for k, v in outs.items():
        print('{} : {}'.format(k, v.size()))
