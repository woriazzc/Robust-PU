import torch
import torch.nn.functional as F


def b_focal_loss(z, labels, weights=None, gamma=1.0, reduction='mean'):
    if weights is None:
        weights = torch.ones(labels.size(0), device=labels.device)
    zero_one_labels = labels.detach().clone()
    zero_one_labels[zero_one_labels == -1] = 0
    probs = torch.sigmoid(z)
    focal_weights = torch.where(labels == 1, torch.pow(1. - probs, gamma), torch.pow(probs, gamma)).detach()
    loss = F.binary_cross_entropy(probs, zero_one_labels.float(), weight=focal_weights, reduction='none')
    if reduction == 'mean':
        loss = torch.mean(loss * weights)
    return loss


def logistic_loss(z, labels):
    return F.softplus(-z * labels)


def sigmoid_loss(z, labels):
    return torch.sigmoid(-z * labels)


def crps(z, labels):
    return torch.sigmoid(-z * labels).pow(2)


def brier(z, labels):
    return 2 * crps(z, labels)


def entropy_loss(z, weights=None, eps=1e-5):
    if weights is None:
        weights = torch.ones(z.size(0), device=z.device)
    probs = torch.sigmoid(z)
    probs = torch.clamp(probs, min=eps, max=1. - eps)
    loss = - probs * torch.log(probs) - (1. - probs) * torch.log(1. - probs)
    loss = torch.mean(loss * weights)
    return loss


def bce_loss(z, labels, weights=None):
    if weights is None:
        weights = torch.ones(labels.size(0), device=labels.device)
    zero_one_labels = labels.detach().clone()
    zero_one_labels[zero_one_labels == -1] = 0
    probs = torch.sigmoid(z)
    loss = F.binary_cross_entropy(probs, zero_one_labels.float(), reduction='none')
    loss = torch.mean(loss * weights)
    return loss


def upu_loss(z, t, prior, weights=None, sur_loss='sigmoid'):
    if weights is None:
        weights = torch.ones(t.size(0), device=t.device)
    if sur_loss == 'sigmoid':
        loss = (lambda x: torch.sigmoid(-x))
    elif sur_loss == 'logistic':
        loss = (lambda x: F.softplus(-x))
    else:
        raise ValueError('Invalid surrogate loss.')
    positive, unlabeled = (t == 1).float(), (t == -1).float()

    n_positive, n_unlabeled = max([1., positive.sum()]), max([1., unlabeled.sum()])

    loss_positive = loss(z)
    loss_unlabeled = loss(-z)  # sigmoid
    positive_risk = torch.sum(prior * positive * weights / n_positive * loss_positive)
    negative_risk = torch.sum(
        (unlabeled * weights / n_unlabeled - prior * positive * weights / n_positive) * loss_unlabeled)
    risk = positive_risk + negative_risk
    return risk


def nnpu_loss(z, t, prior, weights=None, sur_loss='sigmoid', gamma=1.0, beta=0.0):
    """wrapper of loss function for non-negative/unbiased PU learning
        .. math::
            \\begin{array}{lc}
            L_[\\pi E_1[l(f(x))]+\\max(E_X[l(-f(x))]-\\pi E_1[l(-f(x))], \\beta) & {\\rm if nnPU learning}\\\\
            L_[\\pi E_1[l(f(x))]+E_X[l(-f(x))]-\\pi E_1[l(-f(x))] & {\\rm otherwise}
            \\end{array}
    :param z: (Tensor). The shape of ``x`` should be (:math:`N`, ).
    :param t: (Tensor) Target variable for regression. The shape of ``t`` should be (:math:`N`, ).
    :param prior: (float) Constant variable for class prior.
    :param weights: (Tensor)  The shape of ``weight`` should be (:math:`N`, ).
    :param sur_loss: (str) name of surrogate loss function.  The loss function should be non-increasing.
    :param gamma: (float) discount rate in nnpu
    :param beta: (float) tolerance in nnpu
    :return: loss
    """
    if weights is None:
        weights = torch.ones(t.size(0), device=t.device)
    if sur_loss == 'sigmoid':
        loss = (lambda x: torch.sigmoid(-x))
    elif sur_loss == 'logistic':
        loss = (lambda x: F.softplus(-x))
    else:
        raise ValueError('Invalid surrogate loss.')
    positive, unlabeled = (t == 1).float(), (t == -1).float()

    n_positive, n_unlabeled = max([1., positive.sum()]), max([1., unlabeled.sum()])

    y_positive = loss(z).view(-1) * weights
    y_unlabeled = loss(-z).view(-1) * weights
    positive_risk = torch.sum(prior * positive * y_positive / n_positive)
    negative_risk = torch.sum((unlabeled / n_unlabeled - prior * positive / n_positive) * y_unlabeled)
    if negative_risk.data < -beta:
        risk = -gamma * negative_risk
    else:
        risk = positive_risk + negative_risk
    return risk
