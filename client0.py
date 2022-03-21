import socket
import tensorflow as tf
from Functions import *
import time

PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
# SERVER = "10.17.198.243"
SERVER = "192.168.0.6"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect_ex(ADDR)
client.setblocking(0)

print("[TRAINING START] training start...", '\n')
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_u_train = []
y_u_train = []

for i in range(0, len(y_train)):
    if 0 <= y_train[i]<= 4:
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

i = 0
iter_num = 5
while i < iter_num:
    message = weights
    if message:
        send_msg(client, message)
        print(f'number {i} iterations: message send success, sleeping and wait for server response...')
        time.sleep(10)
        if True:
            back_msg = recv_msg(client)
            back_msg = pickle.loads(back_msg)
            print(i, 'recv weights: ', back_msg)
            model_1.set_weights(back_msg)
            model_1.fit(x_1_train, np.array(y_u_train), epochs=6)
            weights1 = model_1.get_weights()
            print(i, 'new_weights: ', weights1, '\n')
            weights = weights1
            i += 1

