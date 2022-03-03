import socket
import pickle
import numpy as np
import sys

HEADER = 64  # Bytes
PORT = 5050
# SERVER = "10.17.198.243"
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECTED"
n = 2  # number of users we need

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

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
    print(new, type(new[0][0]), type(new[0][0][0]))
    return new

# def computing(data):
#     r = []
#     for i in range(0, len(data[0])):
#         result = 0
#         for j in range(0, len(data)):
#             result = result + data[j][i]
#         # r[i] = result/3
#         r.append(result/len(data))
#     return r


def send_back(data):
    data_back = pickle.dumps(data)
    back_length = len(data_back)
    back_length = str(back_length).encode(FORMAT)
    back_length += b' ' * (HEADER - len(back_length))
    client.send(back_length)
    client.send(message)


clients = []
W = []
com = []
server.listen()
print("[STARTING] server is starting...")
print(f"[LISTENING] Server is listening on {SERVER}")

while True:
    # now our endpoint knows about the OTHER endpoint.
    clientsocket, address = server.accept()
    print(f"[NEW CONNECTION] {address} is connected.")

    msg_length = clientsocket.recv(HEADER).decode(FORMAT)
    if msg_length:
        msg_length = int(msg_length)
        message = clientsocket.recv(msg_length)
        msg = pickle.loads(message)
        clients.append(clientsocket)
        W.append(msg)
        clientsocket.send("Msg received".encode(FORMAT))
        print(f"{address} {msg}")
        if len(W) == n:
            com = computing(W)
            print('this is computing result: ', com, '\n')
            # print(clients)
            for client in clients:
                send_back(com)
            # print(W, len(W))
            W.clear()
            clients.clear()
        else:
            pass

    # clientsocket.close()
