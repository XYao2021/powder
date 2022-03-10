import struct
import numpy as np
import pickle

HEADERSIZE = 10  # Bytes
FORMAT = 'utf-8'

def computing(weights):
    new = [[] for i in range(len(weights[0]))]
    for i in range(len(weights[0])):
        for j in range(len(weights[0][i])):
            result = 0
            for k in range(len(weights)):
                # print(c[k][i][j], type(c[k][i][j]))
                result = result + weights[k][i][j]
            new[i].append(result / len(weights))
    for m in range(len(new)):
        new[m] = np.array(new[m])
    # print('this is new_weights: ', new)
    return new

def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    msglen = recvall(sock, 4)
    if not msglen:
        return None
    msg_len = struct.unpack('>I', msglen)[0]
    return recvall(sock, msg_len)


def recvall(sock, message_length):
    recv_data = bytearray()
    while len(recv_data) < message_length:
        packet = sock.recv(message_length - len(recv_data))
        if not packet:
            return None
        recv_data.extend(packet)
    return recv_data
