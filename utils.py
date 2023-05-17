import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

import torch
from torchvision import datasets, transforms


class PUDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)
        assert self.X.size(0) == self.y.size(0)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def preprocess_uci_dataset(dataset_name, data_dir):
    """
    Preprocess UCI datasets
Args:
    :param dataset_name: (str) name of UCI dataset
    :param data_dir:   (str) data dir
Returns:
    :return: X:      (np.array) shape=(N, *), dtype=np.float32
             y:      (np.array) shape=(N), dtype=np.int32
             pos_targets: (list) labels of positive sample
    """
    print(f'Dataset {dataset_name} loading.')
    if dataset_name == 'shuttle':
        train_path = os.path.join(data_dir, 'shuttle.scale.txt')
        test_path = os.path.join(data_dir, 'shuttle.scale.t.txt')
        X_train, y_train = load_svmlight_file(train_path)
        X_train = X_train.toarray()
        X_test, y_test = load_svmlight_file(test_path)
        X_test = X_test.toarray()
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        pos_targets = [1]
    elif dataset_name == 'mushroom':
        file_path = os.path.join(data_dir, 'mushroom.txt')
        X, y = load_svmlight_file(file_path)
        X = X.toarray()
        y[y == 1] = 1
        y[y == 2] = 0
        pos_targets = [1]
        for i in np.unique(y):
            print(f'target={i}: count={(y == i).sum()}')
    elif dataset_name == 'spambase':
        file_path = os.path.join(data_dir, 'spambase.data.txt')
        data = np.loadtxt(file_path, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
        pos_targets = [1]
    else:
        raise ValueError("Invalid dataset name {}".format(dataset_name))

    return X, y, pos_targets


def get_pu_data(X_train, y_train, pos_targets, n_labeled, n_unlabeled, prior):
    """
    construct PU data from raw torchvision.datasets
Args:
    :param X_train: (np.array) shape=(N, *)
    :param y_train: (np.array) shape=(N)
    :param pos_targets: (list) the labels of positive sample
    :param n_labeled:  (int) number of labeled samples
    :param n_unlabeled: (int) number of unlabeled samples
    :param prior:      (float) ratio of unlabeled positive to unlabeled
Returns:
    :return: X_lp:  (np.array) labeled positive data (un-flattened)
             y_lp:  (np.array) labels of X_lp (positive data with label 1)
             X_u:   (np.array) unlabeled data, consists of positive and negative (un-flattened)
             y_u:   (np.array) PN labels of X_u (positive data with label 1, negative data with label NEG)
             X_pre: (np.array) consists of (X_lp, X_u)
             y_pre: (np.array) PU labels of X (labeled data with label 1, unlabeled data with label NEG)
             ty_pre: (np.array) PN labels of X (positive as 1, negative as NEG)
    """
    y_train_bin = np.ones_like(y_train)
    POS, NEG = 1, -1
    pos_mask = torch.tensor([y_train[i] in pos_targets for i in range(y_train.shape[0])])
    neg_mask = torch.tensor([y_train[i] not in pos_targets for i in range(y_train.shape[0])])
    y_train_bin[pos_mask] = POS
    y_train_bin[neg_mask] = NEG
    X, y = np.asarray(X_train, dtype=np.float32), np.asarray(y_train_bin, dtype=np.int32)
    perm = np.random.permutation(y.shape[0])
    X, y = X[perm], y[perm]
    n_positive = (y == POS).sum()  # number of all positive samples (including labeled and unlabeled)
    n_negative = (y == NEG).sum()  # number of all negative samples (all are unlabeled)
    n_lp = n_labeled  # number of labeled samples (all are positive)
    n_u = n_unlabeled  # number of unlabeled samples required
    n_up = int(n_u * prior)  # number of positive samples in unlabeled data
    n_un = n_u - n_up  # number of negative samples in unlabeled data
    assert n_lp + n_up <= n_positive
    assert n_un <= n_negative, f'n_un={n_un}, n_negative={n_negative}'
    X_lp = X[y == POS][:n_lp]  # labeled positive data
    X_up = X[y == POS][n_lp:(n_lp + n_up)]  # unlabeled positive data
    X_un = X[y == NEG][:n_un]  # unlabeled negative data
    X_u = np.asarray(np.concatenate((X_up, X_un), axis=0), dtype=np.float32)
    y_u = np.asarray(np.concatenate((np.ones(n_up), NEG * np.ones(n_un)), axis=0), dtype=np.int32)
    perm = np.random.permutation(y_u.shape[0])
    X_u, y_u = X_u[perm], y_u[perm]
    y_lp = np.ones(n_lp, dtype=np.int32)

    X_pre = np.asarray(np.concatenate((X_lp, X_u), axis=0), dtype=np.float32)
    y_pre = np.asarray(np.concatenate((np.ones_like(y_lp), NEG * np.ones_like(y_u))), dtype=np.int32)
    ty_pre = np.asarray(np.concatenate((y_lp, y_u), axis=0))
    perm = np.random.permutation(y_pre.shape[0])
    X_pre, y_pre, ty_pre = X_pre[perm], y_pre[perm], ty_pre[perm]

    print(
        f'labeled: {n_lp}    unlabeled: {n_u}    unlabeled positive: {n_up}    unlabeled negative: {n_un}    pos_targets: {pos_targets}    prior: {prior}')
    return X_lp, y_lp, X_u, y_u, X_pre, y_pre, ty_pre


def get_datasets(dataset_name, n_labeled, n_unlabeled, prior,
                 n_valid, n_test, root='data', return_pretrain=True):
    """
    construct PU dataloader
Args:
    :param dataset_name:    (str) name of dataset, choices=['cifar10'. 'mnist', 'mushroom', 'shuttle', 'spambase']
    :param n_labeled:       (int) number of labeled samples in training dataset
    :param n_unlabeled:     (int) number of unlabeled samples in training dataset
    :param prior:           (float) ratio of unlabeled positive to unlabeled
    :param n_valid:         (int) number of unlabeled samples in validation dataset
    :param n_test:          (int) number of unlabeled samples in test dataset
    :param root:            (str) dir containing the data file
    :param return_pretrain: (bool) whether return pretrain dataloader
Returns:
    :return: lp_dataset:        (PUDataset) containing labeled positive samples, all labeled 1
             u_dataset:         (PUDataset) containing unlabeled samples w.r.t. prior, positive labeled 1, negative labeled NEG
             pretrain_dataset:    (TensorDataset) return if return_pretrain=True, labeled samples labeled 1, unlabeled samples labeled NEG, and positive as 1, negative as NEG
             val_u_dataset:       (PUDataset) containing validation samples w.r.t. prior, positive labeled 1, negative labeled NEG
             test_u_dataset:      (PUDataset) same as val_set w.r.t. prior, positive labeled 1, negative labeled NEG
             input_size:          (Dataloader) size of flattened data
    """
    if dataset_name == 'cifar10':
        raw_train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transforms.ToTensor())
        raw_test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transforms.ToTensor())
        X_train, y_train = np.asarray(raw_train_dataset.data / 255., dtype=np.float), np.array(raw_train_dataset.targets, dtype=np.int)
        X_test, y_test = np.asarray(raw_test_dataset.data / 255., dtype=np.float), np.array(raw_test_dataset.targets, dtype=np.int)
        X_train, X_test = X_train.transpose((0, 3, 1, 2)), X_test.transpose((0, 3, 1, 2))
        pos_targets = [0, 1, 8, 9]
        print('Dataset CIFAR-10 loaded.')
    elif dataset_name == 'mnist':
        raw_train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor())
        raw_test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transforms.ToTensor())
        X_train, y_train = raw_train_dataset.data / 255., raw_train_dataset.targets
        X_test, y_test = raw_test_dataset.data / 255., raw_test_dataset.targets
        X_train, y_train = X_train.numpy(), y_train.numpy()
        X_test, y_test = X_test.numpy(), y_test.numpy()
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        pos_targets = [0, 2, 4, 6, 8]
        print('Dataset MNIST loaded.')
    else:
        X, y, pos_targets = preprocess_uci_dataset(dataset_name, root)
        perm = np.random.permutation(y.shape[0])
        X, y = X[perm], y[perm]
        if dataset_name == 'shuttle':
            X_train, y_train = X[:-5000], y[:-5000]
            X_test, y_test = X[-5000:], y[-5000:]
        elif dataset_name == 'spambase':
            X_train, y_train = X[:-2000], y[:-2000]
            X_test, y_test = X[-2000:], y[-2000:]
        else:
            X_train, y_train = X[:-3000], y[:-3000]
            X_test, y_test = X[-3000:], y[-3000:]

    X_lp, y_lp, X_u, y_u, X_pre, y_pre, ty_pre = get_pu_data(X_train, y_train, pos_targets, n_labeled, n_unlabeled, prior)
    _, _, val_Xu, val_yu, _, _, _ = get_pu_data(X_train, y_train, pos_targets, 0, n_valid, prior)
    _, _, test_Xu, test_yu, _, _, _ = get_pu_data(X_test, y_test, pos_targets, 0, n_test, prior)

    lp_dataset = PUDataset(X_lp, y_lp)
    u_dataset = PUDataset(X_u, y_u)

    val_u_dataset = PUDataset(val_Xu, val_yu)
    test_u_dataset = PUDataset(test_Xu, test_yu)

    input_size = np.prod(X_u[0].shape)

    print(f'input_size: {input_size}')

    if return_pretrain:
        X_pre = torch.tensor(X_pre)
        y_pre = torch.tensor(y_pre)
        ty_pre = torch.tensor(ty_pre)
        pretrain_dataset = torch.utils.data.TensorDataset(X_pre, y_pre, ty_pre)
        return lp_dataset, u_dataset, pretrain_dataset, val_u_dataset, test_u_dataset, input_size
    else:
        return lp_dataset, u_dataset, val_u_dataset, test_u_dataset, input_size


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adversary(x_ori, y, model, loss_func, alp=0.005):
    x = x_ori.detach()
    x.requires_grad_()
    with torch.enable_grad():
        logits = model(x)
        loss = loss_func(logits, y)
    grad = torch.autograd.grad(loss, [x])[0]
    x = x.detach() + alp * torch.sign(grad.detach())
    x = torch.clamp(x, 0, 1)
    return x
