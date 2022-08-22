import socket
from functions import *
import select


args = args_parser()
HEADER_LENGTH = 10
PORT = args.port
# SERVER = socket.gethostbyname(socket.gethostname())
SERVER = args.server
THRESHOLD = args.bond
PROPAGATION = args.prop

ADDR = (SERVER, PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  #set reuse the address

server.bind(ADDR)
server.listen()
sockets_list = [server]
clients = []
Weights = []

print(f'Listening for connections on {SERVER}...')

com_time = 0
while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    for notified_socket in read_sockets:
        if notified_socket == server:
            client_socket, client_address = server.accept()

            sockets_list.append(client_socket)
            clients.append(client_socket)
            print(f'Accepted new connection from {client_address}...')

            msg_recv = recv_msg(client_socket)
            if msg_recv is False:
                continue
            # print('weights_recv : ', msg_recv[1], '\n')
            Weights.append(msg_recv[1])

            if len(Weights) == THRESHOLD:
                # print('[COMPUTING] start weights computing...')
                new_weights = FedAvg(Weights)
                # print('[NEW WEIGHTS]: ', new_weights, '\n')
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                # print('[FINISHED] new weights already sent back...', '\n')
                print('[NEW FINISHED]', '\n')
                com_time += 1
        else:
            msg_recv = recv_msg(notified_socket)
            if msg_recv is False:
                print(f'Closed connection from: {notified_socket}...')
                sockets_list.remove(notified_socket)
                clients.remove(notified_socket)
                continue
            Weights.append(msg_recv[1])
            # print(f'Message received from {notified_socket}...')
            if len(Weights) == THRESHOLD:
                # print('[COMPUTING] start weights computing...')
                new_weights = FedAvg(Weights)
                # print('[NEW WEIGHTS]: ', new_weights, '\n')
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                # print('[FINISHED] new weights already sent back...', '\n')
                print('[FINISHED]', '\n')
                com_time += 1
                if com_time == PROPAGATION:
                    for user in clients:
                        print(f'[COMMUNICATION COMPLETE] Close connection with {user.getpeername()}')
                        user.close()
                    print('[WAIT FOR NEW CONNECTION]')

    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        clients.remove(notified_socket)
