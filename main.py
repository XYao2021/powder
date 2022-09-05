from timeit import default_timer as timer
import random
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
import copy
from matplotlib import patheffects


args = args_parser()
SERVER = args.server
PORT = args.port

ADDR = (SERVER, PORT)
CLIENT = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
CLIENT.connect(ADDR)

BATCH_SIZE = args.bs
LEARNING_RATE = args.lr
AGGREGATION = args.agg
ITERATION = args.iter * AGGREGATION

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

# org_list = list(np.arange(10))
# targets = random.sample(org_list, args.tsn)
# print('This is target set: ', targets)
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
model_test = copy.deepcopy(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=LEARNING_RATE)
# torch.manual_seed(42)
Train_Loss = []
Test_Loss, Test_Acc = [], []
Comp_time, Trans_time = [], []

for iter in range(1, ITERATION + 1):
    train_begin = timer()
    train_loss, w = train_step(model=model,
                               data_loader=train_dataloader,
                               loss_fn=loss_fn,
                               optimizer=optimizer,
                               accuracy_fn=accuracy_fn,
                               device=device,
                               ITERATION=iter)
    computation_time = timer() - train_begin
    Comp_time.append(computation_time)
    print(iter, 'Computation Time', round(computation_time, 6))
    print('ITERATION: ', iter, '|', 'Train Loss: ', train_loss)

    Train_Loss.append(train_loss)

    send_begin = timer()
    send_msg(CLIENT, ['MSG_CLIENT_TO_SERVER', w])
    new_weights = recv_msg(CLIENT, 'MSG_SERVER_TO_CLIENT')
    transfer_time = timer() - send_begin
    Trans_time.append(round(transfer_time, 6))
    print(iter, 'Communication Time', round(transfer_time, 6))

    model_test.load_state_dict(new_weights[1])

    test_loss, test_acc = test_step(model=model_test,
                                    data_loader=test_dataloader,
                                    loss_fn=loss_fn,
                                    accuracy_fn=accuracy_fn,
                                    device=device,
                                    ITERATION=iter)

    Test_Loss.append(test_loss)
    Test_Acc.append(test_acc)

    if iter % args.iter == 0:
        print('AGGREGATION ', int(iter / args.iter), 'FINISHED')
        print('[NEW WEIGHTS LOAD] on number ', iter, 'iteration', '\n')
        model.load_state_dict(new_weights[1])

print('Train Loss: ', Train_Loss, '\n')
print('Test Loss: ', Test_Loss, '\n')
print('Test Acc: ', Test_Acc, '\n')

print('Comp Time: ', Comp_time, '\n')
print('Trans Time: ', Trans_time, '\n')

print('Average Computation Time: ', 10*(sum(Comp_time)/len(Comp_time)), 10*max(Comp_time), 10*min(Comp_time), '\n')
print('Average Transfer Time: ', 100*(sum(Trans_time)/len(Trans_time)), 100*max(Trans_time), 100*min(Trans_time), '\n')

txt_list = [['Train_Loss: ', Train_Loss],
            ['Test_Loss: ', Test_Loss],
            ['Test_Acc: ', Test_Acc]]

f = open('data_{}.txt'.format(args.ts), 'w')
for item in txt_list:
    f.write("%s\n" % item)

figure, axis = plt.subplots(1, 3)
# For Sine Function
axis[0].plot(range(len(Train_Loss)), Train_Loss, color='red', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
axis[0].set_xlabel("Aggregation")
axis[0].set_ylabel("Train Loss")
axis[0].set_title("Training Loss Function")

axis[1].plot(range(len(Test_Acc)), Test_Acc, color='blue', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
axis[1].set_xlabel("Aggregation")
axis[1].set_ylabel("Test Accuracy")
axis[1].set_title("Test Accuracy Function")

axis[2].plot(range(len(Test_Loss)), Test_Loss, color='green', path_effects=[patheffects.SimpleLineShadow(shadow_color="blue", linewidth=5), patheffects.Normal()])
axis[2].set_xlabel("Aggregation")
axis[2].set_ylabel("Test Loss")
axis[2].set_title(args.ts)

plt.show()

