import os
import numpy as np
import time
import matplotlib.pyplot as plt
import socket


# Proposed_data = np.load('npy_data/proposed_1_50000_2.npy')
# WMMSE_data = np.load('npy_data/WMMSE_1_50000.npy')
# FP_data = np.load('npy_data/FP_1_50000.npy')
# Full_reused_data = np.load('npy_data/full_reused_1_50000.npy')
# Random_data = np.load('npy_data/random_1_50000.npy')
#
#
# plt.plot(Proposed_data[0, :], label='Proposed', color='purple')
# plt.plot(WMMSE_data[0, :], '-.', label='WMMSE', color='orange')
# plt.plot(FP_data[0, :], '--', label='FP', color='blue')
# plt.plot(Full_reused_data[0, :], ':', label='Full Reused', color='brown')
# plt.plot(Random_data[0, :], ':', label='Random', color='green')
#
# max_edge = sorted([max(FP_data[1]), max(WMMSE_data[1]), max(Proposed_data[1])])[1]
# plt.ylim([min(Random_data[2]) - 0.2, max_edge + 0.2])
# plt.xlabel('Slots')
# plt.ylabel('Avg. throughput per BS (bps/Hz)')
# plt.grid()
# plt.legend()
# # plt.savefig('compare_figs/compare_{}.pdf'.format(time.time()))
# plt.show()

# p_mwatt = 10**(p_dbm/10.0)/1000.0

# p_dbm = 10*np.log10(1000)
# p_dbm = 30
# p_watt = 10**(p_dbm/10.0)/1000.0
# print(p_dbm)
# print(p_watt)

# rho = 0
# mu = 1
#
# dc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#
# for d_c in dc:
#     # numerator = (1-(d_c**2)*(1-rho)**2)*(1-rho)**2
#     # denominator = 4*mu**2 + 8*(mu**2)*d_c**2 + (1-(d_c**2)*(1-rho)**2)
#     numerator = (d_c**2)*(1-rho)**2
#     denominator = 3*mu**2 + (d_c**2)*((1-rho)**2 + 3*mu**2)
#     print(numerator / denominator)

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
print(f'The server address is {SERVER}')

