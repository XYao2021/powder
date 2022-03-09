import socket
from Functions import *

PORT = 5050
# SERVER = "10.17.198.243"
SERVER = socket.gethostbyname(socket.gethostname())
# SERVER = "172.16.0.2"
ADDR = (SERVER, PORT)
# DISCONNECT_MESSAGE = "!DISCONNECTED"
n = 2  # number of users we need

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

clients = []
W = []
com = []
server.listen()
print("[STARTING] server is starting...")
print(f"[LISTENING] Server is listening on {SERVER}")

while True:
    clientsocket, address = server.accept()
    print(f'[NEW CONNECTION] connection from {address}...')
    clients.append(clientsocket)
    msg_recv = recv_msg(clientsocket)
    weights_recv = pickle.loads(msg_recv)
    print(f'[RECEIVED] message from {address} has been received')
    W.append(weights_recv)
    # print(len(W), len(clients))
    if len(W) == n:
        print('[COMPUTING] start weights computing...')
        new_weights = computing(W)
        for client in clients:
            send_msg(client, new_weights)
        W.clear()
        clients.clear()
        print('[FINISHED] new weights already sent back...', '\n')

