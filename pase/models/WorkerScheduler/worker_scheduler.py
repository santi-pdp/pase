import torch
import random
import numpy as np
import torch.nn.functional as F
from .min_norm_solvers import MinNormSolver, gradient_normalizers
from torch.autograd import Variable


class backprop_scheduler(object):

    def __init__(self, model, mode=None):
        self.model = model
        self.mode = mode
        self.num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)
        self.Q = torch.zeros(self.num_worker).detach()
        self.last_loss = torch.zeros(self.num_worker).detach()
        self.pi = torch.ones(self.num_worker).detach()


    def __call__(self, preds, label, cls_optim, regr_optim, frontend_optim, device, h=None, dropout_rate=None, delta=None, temperture=None, alpha=None, batch=None):

        if self.mode == "base":
            return self._base_scheduler(preds, label, cls_optim, regr_optim, frontend_optim, device)
        elif self.mode == "adversarial":
            return self._adversarial(preds, label, cls_optim, regr_optim, frontend_optim, device)
        elif self.mode == "select_one":
            return self._select_one(preds, label, cls_optim, regr_optim, frontend_optim, device)
        elif self.mode == "select_half":
            return self._select_half(preds, label, cls_optim, regr_optim, frontend_optim, device)
        elif self.mode == "dropout":
            return self._drop_out(preds, label, cls_optim, regr_optim, frontend_optim, device=device, dropout_rate=dropout_rate)
        elif self.mode == "hyper_volume":
            return self._hyper_volume(preds, label, cls_optim, regr_optim, frontend_optim, device=device, delta=delta)
        elif self.mode == "softmax":
            return self._softmax(preds, label, cls_optim, regr_optim, frontend_optim, temperture=temperture, device=device)
        elif self.mode == "adaptive":
            return self._online_adaptive(preds, label, cls_optim, regr_optim, frontend_optim, temperture=temperture, alpha=alpha, device=device)
        elif self.mode == "MGD":
            return self._MGDA(preds, label, cls_optim, regr_optim, frontend_optim, batch=batch, device=device)
        else:
            raise NotImplementedError

    def _base_scheduler(self, preds, label, cls_optim, regr_optim, frontend_optim, device):
        frontend_optim.zero_grad()
        tot_loss = 0
        losses = {}
        for worker in self.model.classification_workers:
            cls_optim[worker.name].zero_grad()
            loss = worker.loss_weight * worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            tot_loss += loss

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()
            loss = worker.loss_weight * worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            tot_loss += loss

        for worker in self.model.regularizer_workers:
            loss = worker.loss_weight * worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            tot_loss += loss

        tot_loss.backward()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, 1


    def _select_one(self, preds, label, cls_optim, regr_optim, frontend_optim, device):
        self.count += 1
        loss_lst = []
        num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)

        frontend_optim.zero_grad()
        losses = {}

        selected = self.count % num_worker

        # select one
        if selected > 3:
            worker = self.model.classification_workers[selected - 4]
            loss = worker.loss(preds[worker.name], label[worker.name])
        else:
            worker = self.model.classification_workers[selected]
            loss = worker.loss(preds[worker.name], label[worker.name])

        tot_loss = loss

        tot_loss.backward()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, 1

    def _select_half(self, preds, label, cls_optim, regr_optim, frontend_optim, device):
        num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)
        loss_tmp = torch.zeros(num_worker).to(device)
        idx = 0

        frontend_optim.zero_grad()
        losses = {}
        for worker in self.model.classification_workers:
            cls_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        # generate mask
        mask = np.random.randint(2, size=num_worker)
        while np.sum(mask) > 4 or np.sum(mask) < 3:
            mask = np.random.randint(2, size=num_worker)
        mask = torch.from_numpy(mask).type(torch.FloatTensor).to(device)

        #sum up losses
        tot_loss = torch.sum(mask * loss_tmp, dim=0)

        tot_loss.backward()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, 1

    def _drop_out(self, preds, label, cls_optim, regr_optim, frontend_optim, dropout_rate, device):
        loss_tmp = torch.zeros(7, requires_grad=True).to(device)
        idx = 0

        assert dropout_rate is not None
        re_mask = np.random.binomial(1, dropout_rate, size=len(self.model.regression_workers))
        cls_mask = np.random.binomial(1, dropout_rate, size=len(self.model.classification_workers))


        frontend_optim.zero_grad()
        losses = {}
        for i, worker in enumerate(self.model.classification_workers):
            cls_optim[worker.name].zero_grad()

            if cls_mask[i] == 1:
                loss = worker.loss(preds[worker.name], label[worker.name])
            else:
                loss = 0

            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()

            if re_mask[i] == 1:
                loss = worker.loss(preds[worker.name], label[worker.name])
            else:
                loss = 0
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1


        #sum up losses
        tot_loss = torch.sum(loss_tmp, dim=0)

        tot_loss.backward()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, 1

    def _hyper_volume(self, preds, label, cls_optim, regr_optim, frontend_optim, delta ,device):
        assert delta > 1

        num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)
        loss_tmp = torch.zeros(num_worker).to(device)
        idx = 0

        frontend_optim.zero_grad()
        losses = {}
        for worker in self.model.classification_workers:
            cls_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1


        #sum up losses
        eta = delta * torch.max(loss_tmp.detach()).item()
        hyper_votolume = torch.sum(loss_tmp)

        alpha = 1 / (eta - loss_tmp + 1e-6)

        hyper_votolume.backward()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = hyper_votolume

        return losses, alpha

    def _softmax(self, preds, label, cls_optim, regr_optim, frontend_optim, temperture, device):
        assert temperture > 0

        num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)
        loss_tmp = []
        idx = 0

        frontend_optim.zero_grad()
        losses = {}
        for worker in self.model.classification_workers:
            cls_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp.append(loss.item() * temperture)
            # idx += 1

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp.append(loss.item() * temperture)
            # idx += 1


        alpha = self._stable_softmax(loss_tmp)

        tot_loss = 0
        for worker in self.model.classification_workers:
            # tot_loss += alpha[idx] * losses[worker.name]
            tot_loss += losses[worker.name]
            idx += 1
        for worker in self.model.regression_workers:
            # tot_loss += alpha[idx] * losses[worker.name]
            tot_loss += losses[worker.name]
            idx += 1

        # tot_loss = torch.sum(alpha.detach() * loss_vec)
        tot_loss.backward()


        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, alpha

    def _online_adaptive(self, preds, label, cls_optim, regr_optim, frontend_optim, temperture, alpha, device):

        assert temperture > 0 and alpha > 0
        # device = preds['chunk'].device
        num_worker = len(self.model.regression_workers) + len(self.model.classification_workers)
        loss_tmp = torch.zeros(num_worker).to(device)
        idx = 0

        frontend_optim.zero_grad()
        losses = {}
        for worker in self.model.classification_workers:
            cls_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        for worker in self.model.regression_workers:
            regr_optim[worker.name].zero_grad()
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            loss_tmp[idx] = loss
            idx += 1

        R_t = self.last_loss.to(device) - loss_tmp

        with torch.no_grad():
            Q_t = alpha * R_t.detach() + (1 - alpha) * self.Q.to(device)

            self.pi = F.softmax(temperture * Q_t, dim=0)


        tot_loss = torch.sum(loss_tmp)
        tot_loss.backward()

        self.last_loss = loss_tmp.detach()
        self.Q = Q_t.detach()

        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, self.pi

    def _MGDA(self, preds, label, cls_optim, regr_optim, frontend_optim, batch, device):
        frontend_optim.zero_grad()
        losses = {}

        grads = {}
        for worker in self.model.classification_workers:
            self.model.zero_grad()
            h, chunk, preds, labels = self.model.forward(batch, 1, device)
            # print(worker.name)
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            grads[worker.name] = self._get_gen_grads(loss)

        for worker in self.model.regression_workers:
            self.model.zero_grad()
            h, chunk, preds, labels = self.model.forward(batch, 1, device)
            # print(worker.name)
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            grads[worker.name] = self._get_gen_grads(loss)



        sol, min_norm = MinNormSolver.find_min_norm_element([grads[worker].unsqueeze(0) for worker, _ in grads.items()])
        alpha = sol

        tot_loss = 0
        # idx = 0

        self.model.zero_grad()
        h, chunk, preds, labels = self.model.forward(batch, 1, device)
        for worker in self.model.classification_workers:
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            tot_loss += loss
            # tot_loss += sol[idx] * loss


        for worker in self.model.regression_workers:
            loss = worker.loss(preds[worker.name], label[worker.name])
            losses[worker.name] = loss
            tot_loss += loss
            # tot_loss += sol[idx] * loss




        tot_loss.backward()


        for _, optim in cls_optim.items():
            optim.step()

        for _, optim in regr_optim.items():
            optim.step()

        frontend_optim.step()
        losses["total"] = tot_loss

        return losses, alpha

    def _get_gen_grads(self, loss_):
        # grads = torch.autograd.grad(outputs=loss_, inputs=self.model.frontend.parameters())
        self.model.frontend.zero_grad()
        loss_.backward()
        # grads = self.model.frontend.grad()
        for params in self.model.frontend.parameters():

            try:
                grads_ = torch.cat([grads_, params.grad.view(-1)], 0)
            except:
                grads_ = params.grad.view(-1)

        return grads_ / grads_.norm()

    def _stable_softmax(self, x):
        z = np.asarray(x, np.float) - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator / denominator

        return softmax





