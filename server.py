import socket
import pickle
import numpy as np
import sys

HEADER = 10  # Bytes
PORT = 5050
# SERVER = "10.17.198.243"
SERVER = socket.gethostbyname(socket.gethostname())
# SERVER = "172.16.0.2"
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

def recv_all(sock, m_length):
    full_msg = b''
    new_msg = True
    while True:
        message = sock.recv(m_length)
        if new_msg:
            # print("new msg len:", msg[:HEADER])
            msg_len = int(message[:HEADER])
            new_msg = False

        # print(f"full message length: {msg_len}")
        full_msg += msg
        # print(len(full_msg))

        if len(full_msg) - HEADER == m_length:
            return full_msg

def send_back(c, data):
    data_back = pickle.dumps(data)
    back_length = len(data_back)
    back_length = str(back_length).encode(FORMAT)
    back_length += b' ' * (HEADER - len(back_length))
    c.send(back_length)
    c.send(data_back)


clients = []
W = []
com = []
server.listen()
print("[STARTING] server is starting...")
print(f"[LISTENING] Server is listening on {SERVER}")

while True:
    clientsocket, address = server.accept()
    clients.append(clientsocket)

    msg_length = clientsocket.recv(HEADER).decode(FORMAT)
    msg_length = int(msg_length)
    while msg_length:
        # msg = recv_all(clientsocket, msg_length)
        # print('this is msg_recv: ', msg, len(msg))
        # weights_recv = pickle.loads(msg[len(msg) - msg_length:])
        # W.appends(weights_recv)
        # print('present data: ', clients, len(W), msg_length)
        full_msg = b''
        new_msg = True
        while True:
            msg = clientsocket.recv(msg_length)
            if new_msg:
                new_msg = False
            full_msg += msg
            if len(full_msg) - HEADER == msg_length:
                full_msg = b''
                new_msg = True
                break
        print('recv complete: ', len(full_msg), msg_length)
    # while msg_length:
    #     full_msg = b''
    #     new_msg = True
    #     while True:
    #         msg = clientsocket.recv(msg_length)
    #         if new_msg:
    #             # print("new msg len:", msg[:HEADER])
    #             msg_len = int(msg[:HEADER])
    #             new_msg = False
    #
    #         # print(f"full message length: {msg_len}")
    #         full_msg += msg
    #         # print(len(full_msg))
    #
    #         if len(full_msg) - HEADER == msg_length:
    #             print("full msg recvd")
    #             # print(full_msg[HEADER:])
    #             # print(pickle.loads(full_msg[HEADER:]))
    #             new_msg = True
    #             full_msg = b""
    # print(clients, len(W))

    # clientsocket.close()

# while True:
#     # now our endpoint knows about the OTHER endpoint.
#     clientsocket, address = server.accept()
#     print(f"[NEW CONNECTION] {address} is connected.")
#
#     msg_length = clientsocket.recv(HEADER).decode(FORMAT)
#     msg_length = int(msg_length)
#
#     full_msg = b''
#
#     while msg_length:
#         # print(msg_length)
#         message = clientsocket.recv(msg_length)
#         # print(len(message))
#         # print('this is message: ', message)
#         full_msg += message
#         if len(full_msg) == msg_length:
#             msg = pickle.loads(message)
#             # msg = pickle.loads(message)
#             clients.append(clientsocket)
#             W.append(msg)
#             clientsocket.send("Msg received".encode(FORMAT))
#             print(f"{address} {msg}")

#     # clientsocket.close()

