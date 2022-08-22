import time
import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functions import *
import socket


args = args_parser()
SERVER = args.server
PORT = args.port

ADDR = (SERVER, PORT)
CLIENT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
CLIENT.connect(ADDR)

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
MOMENTUM = args.mm
PROPAGATION = args.prop

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.MNIST(root="data",
                            train=True,
                            download=True,
                            transform=ToTensor(),
                            target_transform=None)

test_data = datasets.MNIST(root="data",
                           train=False,
                           download=True,
                           transform=ToTensor(),
                           target_transform=None)

if args.ts != [0, 2, 4, 6, 8, 9]:
    targets = []
    for t in args.ts:
        targets.append(int(t))
else:
    targets = args.ts

train_sets = []
test_sets = []
for i in range(len(train_data.targets)):
    if train_data.targets[i] in targets:
        train_sets.append(train_data[i])
for j in range(len(test_data.targets)):
    if test_data.targets[j] in targets:
        test_sets.append(test_data[j])

# print(len(train_sets), len(test_sets))

train_dataloader = DataLoader(dataset=train_sets,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

torch.manual_seed(42)
model = MNISTModel(input_shape=784,
                   output_shape=10,
                   hidden_units=50)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM)
# torch.manual_seed(42)
Train_Loss, Train_Acc = [], []
Test_Loss, Test_Acc = [], []

for prop in range(PROPAGATION):
    train_loss, train_acc, w = train_step(model=model,
                                          data_loader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          accuracy_fn=accuracy_fn,
                                          device=device,
                                          EPOCH=prop)

    Train_Loss.append(train_loss.item())
    Train_Acc.append(train_acc)

    send_msg(CLIENT, ['MSG_CLIENT_TO_SERVER', w])
    time.sleep(10)

    if True:
        new_weights = recv_msg(CLIENT, 'MSG_SERVER_TO_CLIENT')
        model.load_state_dict(new_weights[1])

        test_loss, test_acc = test_step(model=model,
                                        data_loader=test_dataloader,
                                        loss_fn=loss_fn,
                                        accuracy_fn=accuracy_fn,
                                        device=device,
                                        EPOCH=prop)

        Test_Loss.append(test_loss.item())
        Test_Acc.append(test_acc)

print(f'Train Loss: {Train_Loss}')
print(f'Test Loss: {Test_Loss}')
print(f'Test Acc: {Test_Acc}')

figure, axis = plt.subplots(1, 3)
# For Sine Function
axis[0].plot(range(len(Train_Loss)), Train_Loss)
axis[0].set_xlabel("Propagation")
axis[0].set_ylabel("Train Loss")
axis[0].set_title("Training Loss Function")

axis[1].plot(range(len(Test_Acc)), Test_Acc)
axis[1].set_xlabel("Propagation")
axis[1].set_ylabel("Test Accuracy")
axis[1].set_title("Test Accuracy Function")

axis[2].plot(range(len(Test_Loss)), Test_Loss)
axis[2].set_xlabel("Propagation")
axis[2].set_ylabel("Test Loss")
axis[2].set_title("Test Loss Function")

plt.show()

