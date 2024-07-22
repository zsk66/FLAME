import copy
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sampling import mnist_iid, q_label_skew, dir_label_skew, noise_feature_skew
from sampling import cifar_noniid,dir_quantity_skew,feature_skew, hybrid_skew
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import ImageFolder
from torch.utils.data import TensorDataset
import statistics

from torchvision.transforms import ToTensor
from sklearn.datasets import fetch_covtype
from collections import defaultdict
from typing import Optional, Callable
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity
from torch.utils.model_zoo import tqdm

import pandas as pd
import sys
import os

def get_flat_model_params(model):
    params = []
    for name, param in model.items():
        params.append(param.data.view(-1))
    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to_param_groups(param_groups, flat_params):
    prev_ind = 0
    for name, param in param_groups.items():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size
    return param_groups


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def merge_user_data(user_data_list):
    merged_data = []
    user_data_indices = []

    for user_index, user_data in enumerate(user_data_list):
        merged_data.extend(user_data)
        user_indices = [user_index] * len(user_data)
        user_data_indices.extend(user_indices)

    return merged_data, user_data_indices
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0)
        torch.nn.init.constant_(m.bias, 0)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def c(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.constant_(m.weight, 0)
        torch.nn.init.constant_(m.bias, 0)

def get_dataset(args):
    if args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
        combined_data = ConcatDataset([train_dataset, test_dataset])
    elif args.dataset == 'mmnist':
        data_dir = '../data/mmnist/'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3), (0.3))
        ])

        combined_data = ImageFolder(data_dir, transform=transform)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
        combined_data = ConcatDataset([train_dataset, test_dataset])
    elif args.dataset == 'femnist':
        data_dir = '../data/femnist/'
        dataset = torch.load(data_dir+'femnist.pt')
        combined_data = TensorDataset(dataset[0], dataset[1])
    elif args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)
        combined_data = ConcatDataset([train_dataset, test_dataset])

    if args.iid:
        user_groups = mnist_iid(combined_data, args.num_users)
    elif args.partition == 'dir-label-skew':
        user_groups = dir_label_skew(args.dataset, combined_data, args.num_users)
    elif args.partition == 'feature-skew':
        # only femnist use this partition
        user_groups = feature_skew(dataset, args.num_users)
    elif args.partition == 'q-label-skew':
        user_groups = q_label_skew(args.dataset, combined_data, args.num_users, args.q)
    elif args.partition == 'quality-skew':
        user_groups = noise_feature_skew(args.dataset, combined_data, args.num_users)
    elif args.partition == 'dir-quantity-skew':
        user_groups = dir_quantity_skew(args.dataset, combined_data, args.num_users)
    elif args.partition == 'hybrid-skew':
        user_groups = hybrid_skew(args.dataset, combined_data, args.num_users, args.q)

    return combined_data, user_groups

def average_loss_acc(local_model, num_users, malicious_users):
    benign_users = list(set((range(num_users))) - set(malicious_users))
    num_benign_users = len(benign_users)
    train_loss_personal_local, train_loss_global_local, test_acc_personal_local, test_acc_global_local = [], [], [], []
    test_loss_personal_local, test_loss_global_local = [], []
    train_loss_hybrid_local, test_acc_hybrid_local, test_loss_hybrid_local = [], [], []
    for idx in benign_users:
        train_loss_personal_local.append(local_model[idx].train_personal_loss)
        train_loss_global_local.append(local_model[idx].train_global_loss)
        test_acc_personal_local.append(local_model[idx].test_acc_personal)
        test_acc_global_local.append(local_model[idx].test_acc_global)

        test_loss_personal_local.append(local_model[idx].test_personal_loss)
        test_loss_global_local.append(local_model[idx].test_global_loss)

        test_acc_hybrid_local.append(max(local_model[idx].test_acc_personal, local_model[idx].test_acc_global))
        test_loss_hybrid_local.append(min(local_model[idx].test_personal_loss, local_model[idx].test_global_loss))
        train_loss_hybrid_local.append(min(local_model[idx].train_personal_loss, local_model[idx].train_global_loss))

    train_loss_personal_avg = sum(train_loss_personal_local) / num_benign_users
    train_loss_global_avg = sum(train_loss_global_local) / num_benign_users
    train_loss_hybrid_avg = sum(train_loss_hybrid_local) / num_benign_users

    test_acc_personal_avg = sum(test_acc_personal_local) / num_benign_users
    test_acc_global_avg = sum(test_acc_global_local) / num_benign_users
    test_acc_hybrid_avg = sum(test_acc_hybrid_local) / num_benign_users

    test_loss_personal_avg = sum(test_loss_personal_local) / num_benign_users
    test_loss_global_avg = sum(test_loss_global_local) / num_benign_users
    test_loss_hybrid_avg = sum(test_loss_hybrid_local) / num_benign_users

    test_acc_personal_variance = statistics.variance(test_acc_personal_local)
    test_acc_global_variance = statistics.variance(test_acc_global_local)
    test_acc_hybrid_variance = statistics.variance(test_acc_hybrid_local)



    test_loss_personal_variance = statistics.variance(test_loss_personal_local)
    test_loss_global_variance = statistics.variance(test_loss_global_local)
    test_loss_hybrid_variance = statistics.variance(test_loss_hybrid_local)
    return (train_loss_hybrid_avg,test_acc_hybrid_avg, test_loss_hybrid_avg, train_loss_global_avg, train_loss_personal_avg,
            test_acc_personal_avg, test_acc_global_avg, test_loss_personal_avg, test_loss_global_avg, test_acc_personal_variance,
            test_acc_global_variance,test_loss_personal_variance,test_loss_global_variance, test_acc_hybrid_variance, test_loss_hybrid_variance)

def average_loss_acc_centralized(local_model, num_users, malicious_users):
    train_loss_local, test_acc_local = [], []
    test_loss_local = []
    benign_users = list(set((range(num_users))) - set(malicious_users))
    num_benign_users = len(benign_users)
    for idx in benign_users:
        train_loss_local.append(local_model[idx].train_loss)
        test_acc_local.append(local_model[idx].test_acc)
        test_loss_local.append(local_model[idx].test_loss)

    train_loss_avg = sum(train_loss_local) / num_benign_users
    test_acc_avg = sum(test_acc_local) / num_benign_users
    test_loss_avg = sum(test_loss_local) / num_benign_users
    test_acc_variance = statistics.variance(test_acc_local)
    test_loss_variance = statistics.variance(test_loss_local)
    return train_loss_avg, test_acc_avg, test_loss_avg, test_acc_variance, test_loss_variance
def average_weights(w, num_users, aggr, malicious_frac):
    w_avg = copy.deepcopy(w[0])
    if aggr == 'regular':
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], num_users)
    elif aggr == 'mkrum':
        chosen_solns = copy.deepcopy(w)
        for i in range(0, len(w)):
            chosen_solns[i] = get_flat_model_params(w[i])
        f = int(len(chosen_solns) * malicious_frac)
        dists = torch.zeros(len(chosen_solns), len(chosen_solns))
        scores = torch.zeros(len(chosen_solns))
        for i in range(len(chosen_solns)):
            for j in range(i, len(chosen_solns)):
                dists[i][j] = torch.norm(chosen_solns[i] - chosen_solns[j], p=2)
                dists[j][i] = dists[i][j]
        for i in range(len(chosen_solns)):
            d = dists[i]
            d, _ = d.sort()
            scores[i] = d[:len(chosen_solns) - f - 1].sum()
        tmp_solns = [chosen_solns[i] for i in torch.topk(scores, 5, largest=False).indices]
        stacked_solns = torch.stack(tmp_solns)
        w_avg = torch.mean(stacked_solns, dim=0)
        w_avg = set_flat_params_to_param_groups(w[0], w_avg)
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model              : {args.model}')
    print(f'    Optimizer          : {args.optimizer}')
    print(f'    Framework          : {args.framework}')
    print(f'    Client selection   : {args.strategy}')
    print(f'    Attack             : {args.corrupted}')
    print(f'    Malicious fraction : {args.num_malicious}')

    print(f'    Global Rounds    : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    dataset            : {args.dataset}')
    print(f'    Data partition     : {args.partition}')
    print(f'    Num of users       : {args.num_users}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Learning  Rate     : {args.lr}')
    print(f'    Lambda:            : {args.Lambda}')
    print(f'    rho                : {args.rho}')
    print(f'    mu                 : {args.mu}')
    print(f'    Local Epochs       : {args.local_ep}')
    print(f'    Local Batch size   : {args.local_bs}\n')
    return


def generate_gaussian_matrix(mean, variance, shape):
    gaussian_noise = torch.randn(shape) * torch.sqrt(torch.tensor(variance)) + torch.tensor(mean)
    return gaussian_noise

