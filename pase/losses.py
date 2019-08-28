import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextualizedLoss(object):
    """ With a possible composition of r
        consecutive frames
    """

    def __init__(self, criterion, r=None):
        self.criterion = criterion
        self.r = r

    def contextualize_r(self, tensor):
        if self.r is None:
            return tensor
        assert isinstance(self.r, int), type(self.r)
        # ensure it is a 3-D tensor
        assert len(tensor.shape) == 3, tensor.shape
        # pad tensor in the edges with zeros
        pad_ = F.pad(tensor, (self.r // 2, self.r // 2))
        pt = []
        # Santi:
        # TODO: improve this with some proper transposition and stuff
        # rather than looping, at the expense of more memory I guess
        for t in range(pad_.size(2) - (self.r - 1)):
            chunk = pad_[:, :, t:t+self.r].contiguous().view(pad_.size(0),
                                                             -1).unsqueeze(2)
            pt.append(chunk)
        pt = torch.cat(pt, dim=2)
        return pt

    def __call__(self, pred, gtruth):
        gtruth_r = self.contextualize_r(gtruth)
        loss = self.criterion(pred, gtruth_r)
        return loss


class ZAdversarialLoss(object):

    def __init__(self, z_gen=torch.randn,
                 batch_acum=1,
                 grad_reverse=False,
                 loss='L2'):
        self.z_gen = z_gen
        self.batch_acum = batch_acum
        self.grad_reverse = grad_reverse
        self.loss = loss
        if loss == 'L2':
            self.criterion = nn.MSELoss()
        elif loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Unrecognized loss ', loss)

    def register_DNet(self, Dnet):
        self.Dnet = Dnet

    def forward_grad_reverse(self, step, fake, optim, real,
                             true_lab, fake_lab):
        dreal = self.Dnet(real) 
        dreal_loss = self.criterion(dreal, true_lab)

        dfake = self.Dnet(fake)
        dfake_loss = self.criterion(dfake, fake_lab)
        d_loss = dreal_loss + dfake_loss
        # backprops through whole structure (even G)
        # reversing grads in the D -> G transition
        d_loss.backward(retain_graph=True)
        if step % self.batch_acum == 0:
            # step D optimizer
            optim.step()
            optim.zero_grad()
        # all fake and all real
        return {'afake_loss':dfake_loss,
                'areal_loss':dreal_loss}

    def forward_alternate(self, step, fake, optim, real,
                          true_lab, fake_lab, gfake_exists=False):
        dreal = self.Dnet(real.detach()) 
        dreal_loss = self.criterion(dreal, true_lab)

        dfake = self.Dnet(fake.detach())
        dfake_loss = self.criterion(dfake, fake_lab)
        d_loss = dreal_loss + dfake_loss
        # backprops through D only
        d_loss.backward()
        if step % self.batch_acum == 0:
            # step D optimizer
            optim.step()
            optim.zero_grad()

        greal = self.Dnet(fake)
        greal_loss = self.criterion(greal, true_lab)
        ret_losses = {'dfake_loss':dfake_loss,
                      'dreal_loss':dreal_loss,
                      'd_loss':d_loss,
                      'greal_loss':greal_loss}
        if gfake_exists:
            gfake = self.Dnet(real)
            gfake_loss = self.criterion(gfake, fake_lab)
            g_loss = greal_loss + gfake_loss
            ret_losses['gfake_loss'] = gfake_loss
        else:
            g_loss = greal_loss
        ret_losses['g_loss'] = g_loss
        return ret_losses


    def __call__(self, step, fake, optim, z_true=None,
                 z_true_trainable=False):
        if not hasattr(self, 'Dnet'):
            raise ValueError('Please register Dnet first '
                             'prior to using L2Adversarial Loss.')
        if z_true is None:
            real = self.z_gen(fake.size())
        else:
            real = z_true

        lab_1 = torch.ones(real.shape[0], 1, real.shape[2])
        lab_0 = torch.zeros(lab_1.shape)
        if fake.is_cuda:
            real = real.to('cuda')
            lab_1 = lab_1.to('cuda')
            lab_0 = lab_0.to('cuda')

        if self.grad_reverse:
            losses = self.forward_grad_reverse(step, fake, optim,
                                               z_true, lab_1, lab_0)
        else:
            losses = self.forward_alternate(step, fake, optim,
                                            z_true, lab_1, lab_0,
                                            z_true_trainable)
        return losses

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

if __name__ == '__main__':
    loss = ContextualizedLoss(nn.MSELoss(), r=3)
    pred = torch.randn(1, 3, 5)
    gtruth= torch.randn(1, 1, 5)
    loss(pred, gtruth)
