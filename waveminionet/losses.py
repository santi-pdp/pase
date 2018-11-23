import torch
import torch.nn as nn


class RegressionLoss(object):

    def __call__(self, pred, gtruth):
        loss = self.criterion(pred, gtruth)
        return loss

class AdversarialLoss(object):

    def __init__(self, z_gen=torch.randn,
                 loss='L2'):
        self.z_gen = z_gen
        self.loss = loss
        if loss == 'L2':
            self.criterion = nn.MSELoss()
        elif loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Unrecognized loss ', loss)

    def register_DNet(self, Dnet):
        self.Dnet = Dnet

    def __call__(self, fake, optim):
        if not hasattr(self, 'Dnet'):
            raise ValueError('Please register Dnet first '
                             'prior to using L2Adversarial Loss.')
        optim.zero_grad()
        real = self.z_gen(fake.size())
        if fake.is_cuda:
            real = real.to('cuda')
        dreal = self.Dnet(real)
        lab_1 = torch.ones(dreal.size())
        if fake.is_cuda:
            lab_1 = lab_1.to('cuda')
        dreal_loss = self.criterion(dreal, lab_1)

        dfake = self.Dnet(fake.detach())
        lab_0 = torch.zeros(dfake.size())
        if fake.is_cuda:
            lab_0 = lab_0.to('cuda')
        dfake_loss = self.criterion(dfake, lab_0)
        d_loss = dreal_loss + dfake_loss
        d_loss.backward()
        optim.step()

        greal = self.Dnet(fake)
        greal_loss = self.criterion(greal, lab_1)
        return dreal_loss, dfake_loss, greal_loss

