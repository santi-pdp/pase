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

class WaveAdversarialLoss(nn.Module):

    def __init__(self, discriminator, d_optimizer, size_average=True,
                 loss='L2', batch_acum=1, device='cpu'):
        super().__init__()
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.batch_acum = batch_acum
        if loss == 'L2':
            self.loss = nn.MSELoss(size_average)
            self.labels = [1, -1, 0]
        elif loss == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
            self.labels = [1, 0, 1]
        elif loss == 'Hinge':
            self.loss = None
        else:
            raise ValueError('Urecognized loss: {}'.format(loss))
        self.device = device

    def retrieve_label(self, y, lab_value, name=''):
        #if not hasattr(self, name):
        label = torch.ones(y.size()) * lab_value
        label = label.to(self.device)
        return label
        #    setattr(self, name, label)
        #return getattr(self, name)

    def forward(self, iteration, x_fake, x_real, 
                c_real=None, c_fake=None, grad=True):
        if grad:
            d_real = self.discriminator(x_real, cond=c_real)
            if self.loss:
                rl_lab = self.retrieve_label(d_real, self.labels[0], 'rl_lab')
                d_real_loss = self.loss(d_real, rl_lab)
            else:
                # hinge loss as vanilla GAN with improved objective
                d_real_loss = F.relu(1.0 - d_real).mean()
            
            d_fake = self.discriminator(x_fake.detach(), cond=c_real)
            if self.loss:
                fk_lab = self.retrieve_label(d_fake, self.labels[1], 'fk_lab')
                d_fake_loss = self.loss(d_fake, fk_lab)
            else:
                # hinge loss as vanilla GAN with improved objective
                d_fake_loss = F.relu(1.0 + d_fake).mean()

            if c_fake is not None:
                # an additional label is given to do misalignment signaling
                d_fake_lab = self.discriminator(x_real,
                                                cond=c_fake)
                if self.loss:
                    d_fake_lab_loss = self.loss(d_fake_lab, fk_lab)
                else:
                    d_fake_lab_loss = F.relu(1.0 + d_fake_lab).mean()

                d_loss = d_real_loss + d_fake_loss + d_fake_lab_loss
            else:
                d_loss = d_real_loss + d_fake_loss

            d_loss.backward(retain_graph=True)
            if iteration % self.batch_acum == 0:
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()

        g_real = self.discriminator(x_fake, cond=c_real)
        if self.loss:
            grl_lab = self.retrieve_label(g_real, self.labels[2], 'grl_lab')
            g_real_loss = self.loss(g_real, grl_lab)
        else:
            g_real_loss = - g_real.mean()
        if grad:
            return {'g_loss':g_real_loss, 
                    'd_real_loss':d_real_loss,
                    'd_fake_loss':d_fake_loss}
        else:
            return {'g_loss':g_real_loss}
