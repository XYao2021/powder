import socket
from Functions import *
import select
from options import args_parser

args = args_parser()
HEADER_LENGTH = 10
PORT = 5050
# SERVER = "10.17.198.243"
SERVER = socket.gethostbyname(socket.gethostname())
# SERVER = "127.0.0.1"
# SERVER = "172.16.0.2"
ADDR = (SERVER, PORT)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  #set reuse the address

server.bind(ADDR)
server.listen()
sockets_list = [server]
clients = []
Weights = []
threshold = args.threshold

print('Listening for connections on: ', SERVER, '...')

com_time = 0
while True:
    read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)
    for notified_socket in read_sockets:
        if notified_socket == server:
            client_socket, client_address = server.accept()

            sockets_list.append(client_socket)
            clients.append(client_socket)
            print('Accepted new connection from: ', client_address, '...')

            msg_recv = recv_msg(client_socket)
            if msg_recv is False:
                continue
            # print('weights_recv : ', msg_recv[1], '\n')
            Weights.append(msg_recv[1])

            if len(Weights) == threshold:
                print('[COMPUTING] start weights computing...')
                new_weights = FedAvg(Weights)
                print('[NEW WEIGHTS]: ', new_weights, '\n')
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                print('[FINISHED] new weights already sent back...', '\n')
                com_time += 1
        else:
            msg_recv = recv_msg(notified_socket)
            if msg_recv is False:
                print('Closed connection from: ', notified_socket, '...')
                sockets_list.remove(notified_socket)
                clients.remove(notified_socket)
                continue
            Weights.append(msg_recv[1])
            print('Message received from: ', notified_socket, '...')
            if len(Weights) == threshold:
                print('[COMPUTING] start weights computing...')
                new_weights = FedAvg(Weights)
                print('[NEW WEIGHTS]: ', new_weights, '\n')
                for client in clients:
                    send_msg(client, ['MSG_SERVER_TO_CLIENT', new_weights])
                Weights.clear()
                print('[FINISHED] new weights already sent back...', '\n')
                com_time += 1
                if com_time == args.epochs:
                    for user in clients:
                        print('[COMMUNICATION COMPLETE] Close connection with: ', user.getpeername())
                        user.close()
                    print('[WAIT FOR NEW CONNECTION]')

    for notified_socket in exception_sockets:
        # Remove from list for socket.socket()
        sockets_list.remove(notified_socket)
        clients.remove(notified_socket)
