import math

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

import Metrics
from utils import *


def get_ordered_un_dataset_by_distance(positive_dataset, unlabeled_dataset):
    pos_X, pos_y = positive_dataset.X, positive_dataset.y
    un_X, un_y = unlabeled_dataset.X, unlabeled_dataset.y
    mean_pos_X = pos_X.mean(dim=0)
    distance = Metrics.euclidean_metric(un_X, mean_pos_X)
    order = np.asarray(sorted(range(un_y.size(0)), key=lambda k: distance[k], reverse=True))
    un_X, un_y = un_X[order], un_y[order]
    un_X, un_y = un_X.cpu().numpy(), un_y.cpu().numpy()
    un_dataset = PUDataset(un_X, un_y)
    return un_dataset


def get_ordered_dataset_by_prob(model, dataset, reverse, args):
    training = model.training
    model.eval()
    X, y = dataset.X, dataset.y
    loader = DataLoader(dataset, batch_size=args.batch_size_val, shuffle=False)
    probs = []
    for data, _ in loader:
        if args.cuda:
            data = data.cuda()
        net_out = model(data)
        prob = torch.sigmoid(net_out)
        if prob.dim() == 0:
            prob = prob.reshape(1)
        probs.append(prob.detach())
    probs = torch.cat(probs, dim=0)
    order = np.asarray(sorted(range(y.size(0)), key=lambda k: probs[k], reverse=reverse))
    X, y = X[order], y[order]
    X, y = X.cpu().numpy(), y.cpu().numpy()
    dataset = PUDataset(X, y)
    model.train(training)
    return dataset


class TrainingScheduler:
    def __init__(self, TS_type, init_ratio, max_thresh, grow_steps, p=None, eta=None, lam=0.5):
        super(TrainingScheduler, self).__init__()
        self.init_ratio = init_ratio
        self.type = TS_type
        self.p = p
        self.eta = eta
        self.step = 0
        self.grow_steps = grow_steps
        self.max_thresh = max_thresh
        self.lam = lam
        # assert 0.0 <= self.init_ratio <= self.max_thresh
        self.cal_lib = torch if isinstance(self.init_ratio, torch.Tensor) else math

    def get_next_ratio(self):
        if self.type == 'const':
            ratio = self.init_ratio
        elif self.type == 'linear':
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) / self.grow_steps * self.step
        elif self.type == 'convex':  # from fast to slow
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * self.cal_lib.sin(self.step / self.grow_steps * np.pi * 0.5)
        elif self.type == 'concave':  # from slow to fast
            if self.step > self.grow_steps:
                ratio = self.max_thresh
            else:
                ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * (1. - self.cal_lib.cos(self.step / self.grow_steps * np.pi * 0.5))
        elif self.type == 'exp':
            assert 0 <= self.lam <= 1
            ratio = self.init_ratio + (self.max_thresh - self.init_ratio) * (1. - self.lam ** self.step)
        else:
            raise NotImplementedError(f'Invalid Training Scheduler type {self.type}')

        if self.init_ratio < self.max_thresh:
            ratio = min(ratio, self.max_thresh)
        else:
            ratio = max(ratio, self.max_thresh)
        self.step += 1
        return ratio


def calculate_spl_weights(x, thresh, args, eps=1e-1):
    spl_type = args.spl_type
    assert thresh > 0., 'spl threshold must be positive'
    if spl_type == 'hard':
        # assert 0. <= thresh <= 1.
        # thresh = torch.quantile(x, thresh)
        weights = (x < thresh).float()
    elif spl_type == 'linear':
        weights = 1. - x / thresh
        weights[x >= thresh] = 0.
    elif spl_type == 'log':
        thresh = min(thresh, 1. - eps)
        assert 0. < thresh < 1., 'Logarithmic need thresh in (0, 1)'
        weights = torch.log(x + 1. - thresh) / torch.log(torch.tensor(1. - thresh))
        weights[x >= thresh] = 0.
    elif spl_type == 'mix2':
        gamma = args.mix2_gamma
        weights = gamma * (1. / torch.sqrt(x) - 1. / thresh)
        weights[x <= (thresh * gamma / (thresh + gamma)) ** 2] = 1.
        weights[x >= thresh ** 2] = 0.
    elif spl_type == 'logistic':
        weights = (1. + torch.exp(torch.tensor(-thresh))) / (1. + torch.exp(x - thresh))
    elif spl_type == 'poly':
        t = args.poly_t
        assert t > 1, 't in polynomial must > 1'
        weights = torch.pow(1. - x / thresh, 1. / (t - 1))
        weights[x >= thresh] = 0.
    elif spl_type == 'welsch':
        weights = torch.exp(-x / (thresh * thresh))
    elif spl_type == 'cauchy':
        weights = 1. / (1. + x / (thresh * thresh))
    elif spl_type == 'huber':
        sx = torch.sqrt(x)
        weights = thresh / sx
        weights[sx <= thresh] = 1.
    elif spl_type == 'l1l2':
        # thresh must decrease when using L1L2
        weights = 1. / torch.sqrt(thresh + x)
    else:
        raise ValueError('Invalid spl_type')
    assert weights.min() >= 0. - eps and weights.max() <= 1. + eps, f'weight [{weights.min()}, {weights.max()}] must in range [0., 1.]'
    return weights
