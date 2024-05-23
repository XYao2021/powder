import socket
import select
import argparse

import numpy as np

from MADDPG_Power_Control import MADDPG_Power_Control
from WMMSE import WMMSE_Power_Control
from FP import FP_Power_Control
from utils import *

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-port", help="socket port number", type=int, default=5050)
    parser.add_argument("-threshold", help="number of connected base stations", type=int, default=2)
    parser.add_argument("-bandwidth", type=int, default=220e3)
    parser.add_argument("-method", type=str, default='PROPOSED')
    parser.add_argument("-wmmse_algorithm", type=str, default='WMMSE', help='WMMSE/FULL/RANDOM')
    parser.add_argument("-slots", type=int, default=50000, help='number of slots')


if __name__ == "__main__":
    args = args_parser()
    bandwidth = args.bandwidth
    threshold = args.threshold
    method = args.method
    slots = args.slots

    PORT = args.port
    SERVER = socket.gethostbyname(socket.gethostname())
    print(f'The server address is {SERVER}')
    ADDR = (SERVER, PORT)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind(ADDR)
    server.listen()
    socket_list = [server]
    BSs = []
    UEs = []
    UE_channels = [[] for i in range(args.slots)]

    if method == 'PROPOSED':
        POWER_CONTROL = MADDPG_Power_Control(bandwidth=bandwidth)
    elif method == 'WMMSE':
        POWER_CONTROL = WMMSE_Power_Control(bandwidth=bandwidth, method=args.wmmse_algorithm)
    elif method == 'FP':
        POWER_CONTROL = FP_Power_Control(bandwidth=bandwidth)

    reward_profile = []
    power_profile = []

    index = 0

    while True:
        read_sockets, _, exception_sockets = select.select(socket_list, [], socket_list)
        for notified_socket in read_sockets:
            if notified_socket == server:
                client_socket, client_address = server.accept()
                print(f'Accepted new connection from {client_address}...')

                recv_data = recv_msg(client_socket)
                print(index, type(recv_data[1]), recv_data[1])
                if msg_recv is False:
                    print('[NO MESSAGE]')

                if recv_data[1] == 'UE':
                    UEs.append(client_socket)
                    socket_list.append(client_socket)
                elif recv_data[1] == 'BS':
                    BSs.append(client_socket)
                else:
                    UE_channels[index].append(recv_data[1])

                if len(UE_channels[index]) == threshold:
                    power_profile_slot, reward_profile_slot = POWER_CONTROL.take_action(index=index, channels=np.array(UE_channels[index]))

                    reward_profile.append(reward_profile_slot)
                    power_profile.append(power_profile_slot)

                    for i in range(len(BSs)):
                        send_msg(BSs[i], ['MSG_SERVER_TO_CLIENT', power_profile[index][i]])
                        send_msg(UEs[i], ['MSG_SERVER_TO_CLIENT', 'start receiving'])
                    print('[POWER TRANS FINISHED]')
                    index += 1
            else:
                recv_data = recv_msg(notified_socket)
                print(index, type(recv_data[1]), recv_data[1])
                if msg_recv is False:
                    print(f'Closed connection from: {notified_socket}...')
                    socket_list.remove(notified_socket)
                    UEs.remove(notified_socket)

                UE_channels[index].append(recv_data[1])
                print(f'Message received from {notified_socket}...')
                if len(UE_channels[index]) == threshold:
                    power_profile_slot, reward_profile_slot = POWER_CONTROL.take_action(index=index, channels=np.array(UE_channels[index]))

                    reward_profile.append(reward_profile_slot)
                    power_profile.append(power_profile_slot)

                    for i in range(len(BSs)):
                        send_msg(BSs[i], ['MSG_SERVER_TO_CLIENT', power_profile[index][i]])
                        send_msg(UEs[i], ['MSG_SERVER_TO_CLIENT', 'start receiving'])
                    print('[POWER TRANS FINISHED]')
                    index += 1

        for notified_socket in exception_sockets:
            # Remove from list for socket.socket()
            socket_list.remove(notified_socket)
            UEs.remove(notified_socket)

        if index == slots:
            print('Learning Finished')
            break

    Averaged_rewards = np.mean(reward_profile, axis=1)
    reward_profile = np.array(reward_profile)
    power_profile = np.array(power_profile)
