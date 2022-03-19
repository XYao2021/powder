import socket
import pickle
import numpy as np
import tensorflow as tf
from Functions import *
import select
import errno


HEADER = 10
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
# SERVER = "10.17.198.243"
SERVER = "192.168.0.6"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)
client.setblocking(False)

print("[TRAINING START] training start...")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_u_train = []
y_u_train = []

for i in range(0, len(y_train)):
    if 0 <= y_train[i] <= 4:
        x_u_train.append(x_train[i])
        y_u_train.append(y_train[i])

x_1_train = tf.keras.utils.normalize(x_u_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

"-------------------------Training------------------------------"
model_1 = tf.keras.models.Sequential()
model_1.add(tf.keras.layers.Flatten())  # input layer
model_1.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # output layer

model_1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_1.fit(x_1_train, np.array(y_u_train), epochs=6)

weights = model_1.get_weights()

# send_msg(client, weights)
# # client.close()
# if True:
#     print(f'[WAITING] wait the message from {SERVER}...')
#     msg_back = recv_msg(client)
#     weights_recv = pickle.loads(msg_back)
#     print(weights_recv)
#
# model_1.set_weights(weights_recv)
# print('[NEW_WEIGHTS] new weights has been set...')
#
# model_1.fit(x_1_train, np.array(y_u_train), epochs=6)
# weights1 = model_1.get_weights()

iter_num = 10
for i in range(iter_num):
    send_msg(client, weights)
    if True:
        print(i, f'[WAITING] wait the message from {SERVER}...')
        msg_back = recv_msg(client)
        weights_recv = pickle.loads(msg_back)
        print(weights_recv)

    model_1.set_weights(weights_recv)
    print(i, '[NEW_WEIGHTS] new weights has been set...')

    model_1.fit(x_1_train, np.array(y_u_train), epochs=6)
    weights = model_1.get_weights()
# print(weights1)
# my_username = input("Username: ")
# username = my_username.encode('utf-8')
# username_header = f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')
# client_socket.send(username_header + username)
#
# while True:
#
#     # Wait for user to input a message
#     message = input(f'{my_username} > ')
#
#     # If message is not empty - send it
#     if message:
#
#         # Encode message to bytes, prepare header and convert to bytes, like for username above, then send
#         message = message.encode('utf-8')
#         message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
#         client_socket.send(message_header + message)
#
#     try:
#         # Now we want to loop over received messages (there might be more than one) and print them
#         while True:
#
#             # Receive our "header" containing username length, it's size is defined and constant
#             username_header = client_socket.recv(HEADER_LENGTH)
#
#             # If we received no data, server gracefully closed a connection, for example using socket.close() or socket.shutdown(socket.SHUT_RDWR)
#             if not len(username_header):
#                 print('Connection closed by the server')
#                 sys.exit()
#
#             # Convert header to int value
#             username_length = int(username_header.decode('utf-8').strip())
#
#             # Receive and decode username
#             username = client_socket.recv(username_length).decode('utf-8')
#
#             # Now do the same for message (as we received username, we received whole message, there's no need to check if it has any length)
#             message_header = client_socket.recv(HEADER_LENGTH)
#             message_length = int(message_header.decode('utf-8').strip())
#             message = client_socket.recv(message_length).decode('utf-8')
#
#             # Print message
#             print(f'{username} > {message}')
#
#     except IOError as e:
#         # This is normal on non blocking connections - when there are no incoming data error is going to be raised
#         # Some operating systems will indicate that using AGAIN, and some using WOULDBLOCK error code
#         # We are going to check for both - if one of them - that's expected, means no incoming data, continue as normal
#         # If we got different error code - something happened
#         if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
#             print('Reading error: {}'.format(str(e)))
#             sys.exit()
#
#         # We just did not receive anything
#         continue
#
#     except Exception as e:
#         # Any other exception - something happened, exit
#         print('Reading error: '.format(str(e)))
#         sys.exit()
