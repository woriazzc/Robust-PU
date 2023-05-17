import torch


def accuracy(z, targets):
    return torch.mean((torch.sign(z) == targets).float()).item()


def euclidean_distance(X, v):
    return torch.sum(torch.square(X - v), dim=-1)
