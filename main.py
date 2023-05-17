import copy
import argparse
import pickle
import transformers
import mlflow
import logging

import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import helpers
import models
import lossFunc
from spl_utills import *

global_step = 0
moving_weights_all = None


def pre_train(model, train_loader, val_loader, args):
    lr = args.pre_lr
    epochs = args.pre_epochs
    n_warmup = args.pre_n_warmup
    if args.pre_optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.pre_wd)
    elif args.pre_optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=args.pre_wd)
    else:
        raise ValueError(f'Invalid optimizer name {args.pre_optim}')
    if args.pre_cos:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup,
                                                                 num_training_steps=epochs)
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup)

    history_loss = []
    history_acc = []

    best_val_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.

    for epoch in range(epochs):
        loss_meter = helpers.AverageMeter()
        pu_acc_meter = helpers.AverageMeter()
        pn_acc_meter = helpers.AverageMeter()
        model.train()
        for data, labels, true_labels in train_loader:
            optimizer.zero_grad()
            if args.cuda:
                data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

            net_out = model(data)

            if args.pre_loss == 'bce':
                loss = lossFunc.bce_loss(net_out, labels)
            else:
                loss = getattr(lossFunc, f'{args.pre_loss}_loss')(net_out, labels, args.prior)
            pu_acc = Metrics.accuracy(net_out, labels)
            pn_acc = Metrics.accuracy(net_out, true_labels)

            loss_meter.update(loss.item(), data.size(0))
            pu_acc_meter.update(pu_acc, data.size(0))
            pn_acc_meter.update(pn_acc, data.size(0))

            loss.backward()
            optimizer.step()

        val_loss, val_acc = test(model, val_loader, args)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_model = copy.deepcopy(model.state_dict())

        scheduler.step()

        print(
            f'Pre-Epoch [{epoch + 1} / {epochs}]  Loss: {loss_meter.avg:.5f}    PU-Acc: {pu_acc_meter.avg * 100.0:.5f}  PN-ACC: {pn_acc_meter.avg * 100.0:.5f}    val_loss: {val_loss:.5f}  val_acc: {val_acc * 100.0:.5f}')

        history_loss.append(loss_meter.avg)
        history_acc.append(pu_acc_meter.avg)

    history = {'loss': history_loss, 'acc': history_acc}
    model.load_state_dict(best_val_model)

    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(),
               f'models/{args.dataset}_{args.prior}_{lr}_{epochs}_{args.pre_loss}_{args.pre_batch_size}.pth')
    return history


def test(model, loader, args):
    training = model.training
    model.eval()
    loss_meter = helpers.AverageMeter()
    acc_meter = helpers.AverageMeter()
    with torch.no_grad():
        for data, labels in loader:
            if args.cuda:
                data, labels = data.cuda(), labels.cuda()
            net_out = model(data)
            loss = lossFunc.bce_loss(net_out, labels)
            acc = Metrics.accuracy(net_out, labels)

            loss_meter.update(loss.item(), data.size(0))
            acc_meter.update(acc, data.size(0))

    model.train(training)
    return loss_meter.avg, acc_meter.avg


def get_fea(model, dataloader, args):
    training = model.training
    model.eval()
    fea_all = []
    true_labels_all = []
    labels_all = []
    with torch.no_grad():
        for data, labels, true_labels in dataloader:
            if args.cuda:
                data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

            net_out, fea = model(data, return_fea=True)
            fea_all.append(fea.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
            true_labels_all.append(true_labels.cpu().numpy())
        fea_all = np.concatenate(fea_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        true_labels_all = np.concatenate(true_labels_all, axis=0)
    model.train(training)
    return fea_all, labels_all, true_labels_all


def weighted_dataloader(model, dataloader, thresh_p, thresh_n, args):
    # calculate weights for all
    training = model.training
    model.eval()
    data_all, labels_all, true_labels_all, weights_all, probs_all, fea_all = [], [], [], [], [], []
    global moving_weights_all
    with torch.no_grad():
        for data, labels, true_labels in dataloader:
            if args.cuda:
                data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

            if args.hardness in ['distance', 'cos']:
                net_out, fea = model(data, return_fea=True)
                fea_all.append(fea)
            else:
                net_out = model(data)

            data_all.append(data)
            labels_all.append(labels)
            true_labels_all.append(true_labels)

            # unlabeled data with linear weight
            probs = torch.sigmoid(net_out)
            probs_all.append(probs)

            if args.hardness in ['distance', 'cos']:
                continue

            # loss for calculating unlabeled weight
            if args.hardness == 'logistic':
                loss = lossFunc.logistic_loss(net_out / args.temper_n, -1)
            elif args.hardness == 'sigmoid':
                loss = lossFunc.sigmoid_loss(net_out / args.temper_n, -1)
            elif args.hardness == 'crps':
                loss = lossFunc.crps(net_out / args.temper_n, -1)
            elif args.hardness == 'brier':
                loss = lossFunc.brier(net_out / args.temper_n, -1)
            elif args.hardness == 'focal':
                loss = lossFunc.b_focal_loss(net_out / args.temper_n, -1 * torch.ones_like(labels), gamma=args.focal_gamma, reduction='none')
            else:
                raise ValueError(f'Invalid surrogate loss function {args.hardness}')
            # weights for unlabeled
            weights = calculate_spl_weights(loss.detach(), thresh_n, args)

            # loss for calculating labeled weight
            if args.hardness == 'logistic':
                loss = lossFunc.logistic_loss(net_out / args.temper_p, 1)
            elif args.hardness == 'sigmoid':
                loss = lossFunc.sigmoid_loss(net_out / args.temper_p, 1)
            elif args.hardness == 'crps':
                loss = lossFunc.crps(net_out / args.temper_p, 1)
            elif args.hardness == 'brier':
                loss = lossFunc.brier(net_out / args.temper_p, 1)
            elif args.hardness == 'focal':
                loss = lossFunc.b_focal_loss(net_out / args.temper_p, torch.ones_like(labels), gamma=args.focal_gamma, reduction='none')
            else:
                raise ValueError(f'Invalid hardness function {args.hardness}')
            # weights for labeled
            weights[labels == 1] = calculate_spl_weights(loss[labels == 1].detach(), thresh_p, args)
            weights_all.append(weights)

        data_all = torch.cat(data_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)
        true_labels_all = torch.cat(true_labels_all, dim=0)
        if args.hardness == 'distance':
            fea_all = torch.cat(fea_all, dim=0)
            p_fea = fea_all[labels_all == 1]
            u_fea = fea_all[labels_all == -1]
            p_mean = p_fea.mean(dim=0)
            u_mean = u_fea.mean(dim=0)
            p_dis = Metrics.euclidean_distance(fea_all, p_mean)
            u_dis = Metrics.euclidean_distance(fea_all, u_mean)
            weights_all = torch.where(labels_all == 1, calculate_spl_weights(p_dis / u_dis, thresh_p, args), calculate_spl_weights(u_dis / p_dis, thresh_n, args))
        elif args.hardness == 'cos':
            fea_all = torch.cat(fea_all, dim=0)
            p_fea = fea_all[labels_all == 1]
            u_fea = fea_all[labels_all == -1]
            p_mean = p_fea.mean(dim=0)
            u_mean = u_fea.mean(dim=0)
            p_sim = F.cosine_similarity(fea_all, p_mean)
            u_sim = F.cosine_similarity(fea_all, u_mean)
            weights_all = torch.where(labels_all == 1, calculate_spl_weights(1. - p_sim, thresh_p, args), calculate_spl_weights(1 - u_sim, thresh_n, args))
        else:
            weights_all = torch.cat(weights_all, dim=0)
        if moving_weights_all is None:
            moving_weights_all = weights_all
        else:
            moving_weights_all = args.phi * moving_weights_all + (1. - args.phi) * weights_all
        probs_all = torch.cat(probs_all, dim=0)

        unlabel_weights = moving_weights_all[labels_all == -1]
        unlabel_true_labels = true_labels_all[labels_all == -1]
        unlabel_probs = probs_all[labels_all == -1]
        logging.info(
            f'Mean weight of labeled: {weights_all[labels_all == 1].mean()}\tMean weight of negative-unlabeled: {unlabel_weights[unlabel_true_labels == -1].mean()}')
        logging.info(
            f'Mean probability of labeled:  {probs_all[labels_all == 1].mean()}\tMean probability of positive-unlabeled: {unlabel_probs[unlabel_true_labels == 1].mean()}\tMean probability of negative-unlabeled: {unlabel_probs[unlabel_true_labels == -1].mean()}')

    dataloader = DataLoader(TensorDataset(data_all, labels_all, true_labels_all, moving_weights_all), shuffle=True,
                            batch_size=args.batch_size)
    model.train(training)
    return dataloader


def train_episode(model, dataloader, args):
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.cos:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup,
                                                                 num_training_steps=args.inner_epochs)
    else:
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.n_warmup)

    # Test on training set before training
    model.eval()
    with torch.no_grad():
        meter = helpers.AverageMeter()
        for data, labels, true_labels, weights in dataloader:
            optimizer.zero_grad()
            if args.cuda:
                data, labels, true_labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

            net_out = model(data)

            # loss w.r.t. pseudo labels
            if args.loss == 'bce':
                loss = lossFunc.bce_loss(net_out, labels, weights)
            elif args.loss == 'focal':
                loss = lossFunc.b_focal_loss(net_out, labels, weights, gamma=args.focal_gamma)
            else:
                loss = getattr(lossFunc, f'{args.loss}_loss')(net_out, labels, args.prior, weights)
            meter.update(loss.item(), labels.size(0))
        logging.info(f'Loss before training: {meter.avg}')

    if args.restart:
        model.reset_para()

    model.train()

    tot_loss_meter = helpers.AverageMeter()
    tot_true_loss_meter = helpers.AverageMeter()
    tot_acc_meter = helpers.AverageMeter()
    for inner_epoch in range(args.inner_epochs):
        loss_meter = helpers.AverageMeter()
        for data, labels, true_labels, weights in dataloader:
            optimizer.zero_grad()
            if args.cuda:
                data, labels, true_labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

            net_out = model(data)

            # loss w.r.t. pseudo labels
            if args.loss == 'bce':
                loss = lossFunc.bce_loss(net_out, labels, weights)
                if args.debug:
                    net_out_u = net_out[labels == -1]
                    true_labels_u = true_labels[labels == -1]
                    loss_up = getattr(lossFunc, f'{args.hardness}_loss')(net_out_u[true_labels_u == 1] / args.temper_n, -1)
                    loss_un = getattr(lossFunc, f'{args.hardness}_loss')(net_out_u[true_labels_u == -1] / args.temper_n, -1)
                    loss_p = getattr(lossFunc, f'{args.hardness}_loss')(net_out[labels == 1] / args.temper_p, 1)
                    global global_step
                    mlflow.log_metric('loss_up', loss_up.mean().detach().cpu().numpy(), global_step)
                    mlflow.log_metric('loss_un', loss_un.mean().detach().cpu().numpy(), global_step)
                    mlflow.log_metric('loss_p', loss_p.mean().detach().cpu().numpy(), global_step)
                    global_step += 1
            elif args.loss == 'focal':
                loss = lossFunc.b_focal_loss(net_out, labels, weights, gamma=args.focal_gamma)
            else:
                loss = getattr(lossFunc, f'{args.loss}_loss')(net_out, labels, args.prior, weights)

            # loss w.r.t. true labels
            true_loss = lossFunc.bce_loss(net_out, true_labels, weights)
            # acc w.r.t. true labels
            acc = Metrics.accuracy(net_out, true_labels)

            tot_loss_meter.update(loss.item(), data.size(0))
            tot_acc_meter.update(acc, data.size(0))
            tot_true_loss_meter.update(true_loss.item(), data.size(0))

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), labels.size(0))

        scheduler.step()
        logging.debug(f'inner epoch [{inner_epoch + 1} / {args.inner_epochs}]  train loss: {loss_meter.avg}')
        # if args.debug:
        #     global global_step
        #     mlflow.log_metric('train_loss', loss_meter.avg, global_step)
        #     global_step += 1

    return tot_loss_meter.avg, tot_acc_meter.avg, tot_true_loss_meter.avg


def train(model, positive_dataset, unlabeled_dataset, val_dataset, test_dataset, args):
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)

    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    positive_data, positive_labels = positive_dataset.X, positive_dataset.y
    unlabeled_data, unlabeled_labels = unlabeled_dataset.X, unlabeled_dataset.y

    # for SPL
    cl_scheduler_p = TrainingScheduler(args.scheduler_type_p, args.alpha_p, args.max_thresh_p, args.grow_steps_p, args.p,
                                       args.eta)
    cl_scheduler_n = TrainingScheduler(args.scheduler_type_n, args.alpha_n, args.max_thresh_n, args.grow_steps_n, args.p,
                                       args.eta)

    history_loss = []
    history_acc = []
    history_true_loss = []
    history_val_loss = []
    history_val_acc = []

    val_best_acc = 0.
    val_best_index = -1
    val_best_model = copy.deepcopy(model.state_dict())

    fea_all = []

    for episode in range(epochs):
        # get next lambda
        thresh_p = cl_scheduler_p.get_next_ratio()
        thresh_n = cl_scheduler_n.get_next_ratio()
        helpers.prRedWhite(f'thresh_p = {thresh_p:.3f}  thresh_n = {thresh_n:.3f}')
        cur_data = torch.cat((positive_data, unlabeled_data), dim=0)
        cur_labels = torch.cat((positive_labels, -torch.ones_like(unlabeled_labels)), dim=0)
        cur_true_labels = torch.cat((positive_labels, unlabeled_labels), dim=0)
        perm = np.random.permutation(cur_data.size(0))
        cur_data, cur_labels, cur_true_labels = cur_data[perm], cur_labels[perm], cur_true_labels[perm]
        cur_loader = DataLoader(TensorDataset(cur_data, cur_labels, cur_true_labels), batch_size=batch_size,
                                shuffle=True)
        weighted_loader = weighted_dataloader(model, cur_loader, thresh_p, thresh_n, args)

        if args.vis and episode == 0:
            fea_all.append(get_fea(model, cur_loader, args))

        tot_loss, tot_acc, tot_true_loss = train_episode(model, weighted_loader, args)

        if args.vis:
            fea_all.append(get_fea(model, cur_loader, args))

        val_loss, val_acc = test(model, val_loader, args)
        test_loss, test_acc = test(model, test_loader, args)
        print(
            f'Episode [{episode + 1} / {epochs}]   Pseudo_Loss: {tot_loss:.5f}  True_Loss: {tot_true_loss:.5f}  True_Acc: {tot_acc * 100.0:.5f}   val_loss: {val_loss:.5f}  val_acc: {val_acc * 100.0:.5f}    test_loss: {test_loss: .5f}  test_acc: {test_acc * 100.0:.5f}')

        history_loss.append(tot_loss)
        history_acc.append(tot_acc)
        history_true_loss.append(tot_true_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        if args.debug:
            mlflow.log_metric('val_loss', val_loss)
            mlflow.log_metric('val_err', 100.0 - val_acc * 100.0)

        # Early stop
        if val_acc > val_best_acc:
            val_best_acc = val_acc
            val_best_index = episode
            val_best_model = copy.deepcopy(model.state_dict())
        else:
            if episode - val_best_index >= patience:
                print(f'=== Break at epoch {val_best_index + 1} ===')
                fea_all = fea_all[:val_best_index + 2]
                break

    model.load_state_dict(val_best_model)

    if args.vis:
        if not os.path.exists('data_anal'):
            os.mkdir('data_anal')
        pickle.dump(fea_all, open(f'data_anal/{args.dataset}_{args.prior}{args.ed}.npy', 'wb'))

    history = {'pseudo_loss': history_loss, 'true_loss': history_true_loss, 'acc': history_acc,
               'val_loss': history_val_loss, 'val_acc': history_val_acc}

    return history


def prepare_and_run(args):
    seed_all(args.seed)

    positive_dataset, unlabeled_dataset, pretrain_dataset, val_dataset, test_dataset, input_size = get_datasets(
        args.dataset,
        args.n_labeled,
        args.n_unlabeled,
        args.prior,
        root=args.data_dir,
        n_valid=args.n_valid,
        n_test=args.n_test,
        return_pretrain=True)
    args.input_size = input_size

    model = getattr(models, args.model)(args.input_size)
    if args.cuda:
        model = model.cuda()

    if args.pretrained:
        print(f'Model loaded from: {args.pretrained}.')
        model.load_state_dict(torch.load(args.pretrained))
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.pre_batch_size, shuffle=True)
        pre_train(model, pretrain_loader, val_loader, args)

    seed_all(args.seed)

    # Test before train
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_val, shuffle=False)
    val_loss, val_acc = test(model, val_loader, args)
    un_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size_val, shuffle=False)
    un_loss, un_acc = test(model, un_loader, args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_loss, test_acc = test(model, test_loader, args)
    helpers.prYellow(
        f'Before training  un-Loss: {un_loss:.5f}  un-Acc: {un_acc * 100.0: .5f}    val-Loss: {val_loss:.5f}  val-Acc: {val_acc * 100.0: .5f}   Test-Acc: {test_acc * 100.0:.5f}')

    history = train(model, positive_dataset, unlabeled_dataset, val_dataset, test_dataset, args)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
    test_loss, test_acc = test(model, test_loader, args)
    test_err = 1. - test_acc
    print(f'Test    Loss: {test_loss}   Error: {100.0 * test_err}')

    return history, test_loss, test_err


def main():
    args = parser.parse_args()
    args.cuda = (not args.no_cuda)
    logging.basicConfig(level=(logging.DEBUG if args.debug else logging.INFO))
    if args.run_all:
        args.debug = False

    if args.dataset == 'cifar10':
        args.model = 'CNN'
        args.n_labeled = 2000
        args.n_unlabeled = 4000
        args.n_valid = 500
        args.n_test = 5000
    elif args.dataset == 'mnist':
        args.model = 'normalNN'
        args.n_labeled = 2000
        args.n_unlabeled = 4000
        args.n_valid = 500
        args.n_test = 5000
    # for UCI datasets
    else:
        args.model = 'normalNN'
        args.n_labeled = 400
        args.n_unlabeled = 800
        args.n_valid = 100
        args.n_test = 1000

    print(args)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.run_all:

        for prior in [0.2, 0.4, 0.6]:
            args.prior = prior
            test_errors = []

            for seed in range(10):
                args.seed = seed
                history, test_loss, test_err = prepare_and_run(args)

                test_errors.append(test_err)

            test_errors = np.array(test_errors)

            mlflow.log_metric(f'test_err{prior}', test_errors.mean() * 100.0)

            print(f'--- prior: {args.prior}   Error: {test_errors.mean() * 100.0:.3f} ({test_errors.std():.3f}) ---')

    else:
        prepare_and_run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Implementation of CL+PU')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--ed', default='')
    # Dirs
    parser.add_argument("--output_dir", type=str, default=os.getenv("AMLT_OUTPUT_DIR", "results"),
                        help='output dir (default: results)')
    parser.add_argument("--data_dir", type=str, default=os.getenv("AMLT_DATA_DIR", "data"),
                        help="Directory where dataset is stored")
    # Dataset
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'mnist', 'mushroom', 'shuttle', 'spambase'],
                        default='mnist',
                        help='dataset name (default: mnist)')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

    parser.add_argument('--n_labeled', type=int, default=2000, help='number of positive samples (default: 2000)')
    parser.add_argument('--n_unlabeled', type=int, default=4000, help='number of unlabeled samples (default: 4000)')
    parser.add_argument('--n_valid', type=int, default=500, help='number of valid samples (default: 500)')
    parser.add_argument('--n_test', type=int, default=5000, help='number of valid samples (default: 5000)')
    parser.add_argument('--prior', type=float, default=0.2,
                        help='ratio of unlabeled positive to unlabeled (default 0.2)')

    # GPU
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda (default: False)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='set gpu id to use (default: 0)')

    # Pre-training
    parser.add_argument('--pre_optim', type=str, default='adam', choices=['adam', 'sgd'],
                        help='name of optimizer for pre-training (default adam)')
    parser.add_argument('--pre_epochs', type=int, default=400,
                        help='number of pre-training epochs (default 400)')
    parser.add_argument('--pre_lr', type=float, default=1e-3, help='pre-training learning rate (default 1e-3)')
    parser.add_argument('--pre_wd', type=float, default=0., help='weight decay for pre-training (default 0.)')
    parser.add_argument('--pre_batch_size', type=int, default=64, help='batch size for pre-training (default 64)')
    parser.add_argument('--pre_n_warmup', type=int, default=0,
                        help='number of warm-up steps in pre-training (default 0)')
    parser.add_argument('--pre_cos', action='store_true',
                        help='Use cosine lr scheduler in pre-training (default False)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='pre-trained model path (default None)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs to run (default: 100)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--batch_size_val', default=128, type=int,
                        help='mini-batch size of validation (default: 128)')
    parser.add_argument('--optim', type=str, default='adam', help='type of optimizer for training (default: adam)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', default=0., type=float, help='weight decay (default 0.)')
    parser.add_argument('--decay_epoch', default=-1, type=int,
                        help='Reduces the learning rate every decay_epoch (default -1)')
    parser.add_argument('--lr_decay', default=0.5, type=float,
                        help='Learning rate decay for training (default: 0.5)')
    parser.add_argument('--cos', action='store_true',
                        help='Use cosine lr scheduler (default False)')
    parser.add_argument('--n_warmup', default=0, type=int,
                        help='Number of warm-up steps (default: 0)')
    parser.add_argument('--patience', default=5, type=int, help='patience for early stopping (default 5)')
    parser.add_argument('--restart', action='store_true',
                        help='reset model before training in each episode (default: False)')

    # Test
    parser.add_argument('--run_all', action='store_true', help='run all experiences with 20 seeds (default False)')

    # CL
    parser.add_argument('--inner_epochs', type=int, default=1,
                        help='number of epochs to run after each dataset update (default: 1)')
    parser.add_argument('--max_thresh_p', type=float, default=2., help='maximum of threshold for labeled (default 2.0)')
    parser.add_argument('--max_thresh_n', type=float, default=2., help='maximum of threshold for unlabeled (default 2.0)')
    parser.add_argument('--grow_steps_p', type=int, default=10, help='number of step to grow to max_thresh for labeled (default 10)')
    parser.add_argument('--grow_steps_n', type=int, default=10, help='number of step to grow to max_thresh for unlabeled (default 10)')
    parser.add_argument('--scheduler_type_p', type=str, default='linear',
                        help='type of training scheduler for labeled (default linear)')
    parser.add_argument('--scheduler_type_n', type=str, default='linear',
                        help='type of training scheduler for unlabeled (default linear)')
    parser.add_argument('--alpha_p', type=float, default=0.1, help='initial threshold for labeled (default 0.1)')
    parser.add_argument('--alpha_n', type=float, default=0.1, help='initial threshold for unlabeled (default 0.1)')
    parser.add_argument('--eta', type=float, default=1.1,
                        help='alpha *= eta in each step for scheduler exp (default 1.1)')
    parser.add_argument('--p', type=int, default=2, help='p for scheduler root-p (default 2)')
    parser.add_argument('--spl_type', type=str, default='welsch',
                        choices=['hard', 'linear', 'log', 'mix2', 'logistic', 'poly', 'welsch', 'cauchy', 'huber',
                                 'l1l2'],
                        help='type of soft sp-regularizer (default welsch)')
    parser.add_argument('--mix2_gamma', type=float, default=1.0, help='gamma in mixture2 (default 1.0)')
    parser.add_argument('--poly_t', type=int, default=3, help='t in polynomial (default 3)')

    # PU
    parser.add_argument('--pre_loss', type=str, default='nnpu', choices=['bce', 'nnpu', 'upu'])
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'nnpu', 'upu', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='gamma for focal loss')
    parser.add_argument('--hardness', type=str, default='logistic',
                        help='hardness function used to calculate weights (default: logistic)')
    parser.add_argument('--temper_n', type=float, default=1.0, help='temperature to smooth logits for unlabeled (default: 1.0)')
    parser.add_argument('--temper_p', type=float, default=1.0, help='temperature to smooth logits for labeled (default: 1.0)')
    parser.add_argument('--phi', type=float, default=0., help='momentum for weight moving average (default: 0.)')

    main()
