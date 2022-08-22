import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import struct
import numpy as np
import pickle
import copy
import socket
import argparse


HEADERSIZE = 10  # Bytes
FORMAT = 'utf-8'
SIZE = 1024

def args_parser():
    parse = argparse.ArgumentParser()

    parse.add_argument('-prop', type=int, default=100, help='Global Propagation times')
    parse.add_argument('-lr', type=float, default=0.1, help='Learning Rate of the Model')
    parse.add_argument('-mm', type=float, default=0.5, help='Momentum for SGD algorithm')
    parse.add_argument('-bs', type=int, default=32, help='Batch Size for model')
    parse.add_argument('-ts', type=list, default=[0, 2, 4, 6, 8, 9], help='Target set for training and local testing')

    parse.add_argument('-server', type=str, default='172.16.0.1', help='Server IP address')
    parse.add_argument('-port', type=int, default=5050, help='Socket port')
    parse.add_argument('-bond', type=int, default=2, help='Threshold for FedAvg on Sever side')

    args = parse.parse_args()
    return args

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

class MNISTModel(nn.Module):  # Improve the model V0 with nonlinear activation function nn.Relu()
    def __init__(self, input_shape,
                 output_shape,
                 hidden_units):
        super().__init__()
        self.layer_stack = nn.Sequential(nn.Flatten(),  # Equal to x.view(-1, 784)
                                         nn.Linear(in_features=input_shape, out_features=hidden_units),
                                         nn.ReLU(),
                                         nn.Linear(in_features=hidden_units, out_features=output_shape),
                                         nn.ReLU())

    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)

def eval_model(model,
               data_loader,
               loss_fn,
               accuracy_fn):

    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y, y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}

def train_step(model,
               data_loader,
               loss_fn,
               optimizer,
               accuracy_fn,
               device,
               EPOCH):

    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    w = model.state_dict()
    print(EPOCH, 'Train Loss: ', train_loss)
    print(EPOCH, 'Train Acc: 'train_acc, '\n')
    return train_loss, train_acc, w

def test_step(model,
              data_loader,
              loss_fn,
              accuracy_fn,
              device,
              EPOCH):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y, test_pred.argmax(dim=1))
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(EPOCH, 'Test Loss: ', test_loss)
        print(EPOCH, 'Test Acc: ', test_acc, '\n')
    return test_loss, test_acc

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.send(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername(), '\n')

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername(), '\n')

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg

