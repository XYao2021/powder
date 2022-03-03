import socket
import pickle
import numpy as np
import tensorflow as tf
import time

HEADER = 64
PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
# SERVER = "10.17.198.243"
SERVER = "10.17.198.243"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

print("[TRAINING START] training start...")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_u_train = []
y_u_train = []

for i in range(0, len(y_train)):
    if y_train[i] >= 5:
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
# weights = np.array([1.0111, 2.0222, 3.0333, 4.0444, 5.0555, 6.06666])

def send(msg):
    message = pickle.dumps(msg)
    msg_length = len(message)
    send_length = str(msg_length).encode(FORMAT)
    send_length += b' ' * (HEADER - len(send_length))
    client.send(send_length)
    client.send(message)
    print(client.recv(2048).decode(FORMAT))


send(weights)

msg_length_back = client.recv(HEADER).decode(FORMAT)
if msg_length_back:
    msg_length_back = int(msg_length_back)
    message_back = client.recv(msg_length_back)
    msg_back = pickle.loads(message_back)
    print(msg_back, '\n', '[COMPLETE] transmission complete...')
