#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import random
import socket
import time
import pickle

from Functions import *
from sampling import mnist_iid, mnist_noniid, cifar_iid
from options import args_parser
from Update import LocalUpdate
from Nets import MLP, CNNMnist, CNNCifar
from test import test_img

# matplotlib.use('Agg')  # XY: Using Agg(for no GPU condition) mode, just save image, cannot plot it
if __name__ == '__main__':
    # socket info
    args = args_parser()
    PORT = 5050
    SERVER = args.S
    ADDR = (SERVER, PORT)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)

    # parse args
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    # XY: Specific Training and Test dataset and shuffled Training sets idx
    training_list = list(dataset_train.targets)
    test_list = list(dataset_test.targets)

    if args.target_set != [0, 1, 2, 3, 4]:
        targets = []
        for t in args.target_set:
            targets.append(int(t))
    else:
        targets = args.target_set

    new_dataset_train = []
    new_dataset_test = []
    for target in training_list:
        if target in targets:
            new_dataset_train.append(dataset_train[training_list.index(target)])
    for item in test_list:
        if item in targets:
            new_dataset_test.append(dataset_test[test_list.index(item)])
    idx_len = len(new_dataset_train) - len(new_dataset_train) % 10
    idx = list(np.arange(len(new_dataset_train)))
    random.shuffle(idx)
    i_d = np.arange(0, idx_len, dtype=int)

    img_size = dataset_train[0][0].shape
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':  # Multi-Layer Perceptron 多层感应器
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # XY: Turn to Training mode, activate dropout layer and the same layers
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training  XY: Set all parameters to 0 and None for next training process
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_acc, test_loss = [], []

    for iter in range(args.epochs):
        print('[START] Start', iter, 'iteration local update ...')
        loss_locals = []
        local = LocalUpdate(args=args, dataset=new_dataset_train, idxs=i_d)  # XY: Each Local update has 10 local iterations
        w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

        print('Round {}, Average loss {:.8f}'.format(iter, loss), '\n')
        loss_train.append(loss)  # Print the loss trend changing along with iteration times

        send_msg(client, ['MSG_CLIENT_TO_SERVER', w])
        print('[WAITING] number', iter, 'iterations finished and waiting for server response ...')
        time.sleep(15)
        if True:
            # back_msg = recv_msg(client)
            back_msg = recv_msg(client, 'MSG_SERVER_TO_CLIENT')
            print('[RECEIVED] Received new weights from server and load weights then start next iteration ... ')
            net_glob.load_state_dict(back_msg[1])
            # XY: Test the model  Original test function cannot be running because Mac don't have GPU mode for cuda (torch.cuda.is_available() = False)
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            test_acc.append(acc_test)
            test_loss.append(loss_test)
            print(iter, '[TEST RESULT]: ', acc_test, loss_test, '\n')

    print('loss_train: ', loss_train, '\n')
    print('test_acc: ', test_acc, '\n')
    print('test_loss: ', test_loss, '\n')
    # Plot loss curve  XY: just save the raster image without plot it
    # figure, axis = plt.subplots(1, 3)
    # # For Sine Function
    # axis[0].plot(range(len(loss_train)), loss_train)
    # axis[0].set_xlabel("Epochs")
    # axis[0].set_ylabel("Train Loss")
    # axis[0].set_title("Training Loss Function")
    #
    # axis[1].plot(range(len(test_acc)), test_acc)
    # axis[1].set_xlabel("Epochs")
    # axis[1].set_ylabel("Test Accuracy")
    # axis[1].set_title("Test Accuracy Function")
    #
    # axis[2].plot(range(len(test_loss)), test_loss)
    # axis[2].set_xlabel("Epochs")
    # axis[2].set_ylabel("Test Loss")
    # axis[2].set_title("Test Loss Function")
    #
    # plt.show()
