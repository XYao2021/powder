import numpy as np
import os
import time
from config import *
from utils import *
import matplotlib.pyplot as plt


def WMMSE(num_iters, channel_map, power_max, noise, weights):

    channel_map = np.array(channel_map)
    V_map = np.zeros([NUM_BS, num_iters])
    sched_UEs = [0 for i in range(NUM_BS)]  # Scheduled UE index
    # print(power_max, np.sqrt(power_max))

    "Initialize U, V, W"
    V = np.sqrt(0.5) * np.sqrt(power_max) * np.ones(NUM_BS)  # Initial power = 0.5 * power_max ???
    # V = np.zeros(NUM_BS)

    U = np.ones(NUM_BS)
    for bs in range(NUM_BS):
        ue_sched_g = bs * UE_PER_BS + sched_UEs[bs]
        numerator = np.sqrt(channel_map[ue_sched_g, bs]) * V[bs]
        interference = [channel_map[ue_sched_g, i] * V[i]**2 for i in range(NUM_BS)]
        denomiator = sum(interference) + noise
        U[bs] = numerator / denomiator

    W = np.zeros(NUM_BS)
    for bs in range(NUM_BS):
        ue_sched_g = bs * UE_PER_BS + sched_UEs[bs]
        E_i = 1.0 - U[bs] * np.sqrt(channel_map[ue_sched_g, bs]) * V[bs]
        W[bs] = 1 / max(1e-35, E_i)

    "Iterative update U, V, W"
    for i in range(num_iters):
        "Update V and V_map (record V values for all iterations)"
        for bs in range(NUM_BS):
            ue_sched_g = bs * UE_PER_BS + sched_UEs[bs]
            v_numerator = weights[bs] * W[bs] * U[bs] * np.sqrt(channel_map[ue_sched_g, bs])
            interference = [weights[j] * W[j] * U[j]**2 * channel_map[j * UE_PER_BS + sched_UEs[j], bs] for j in range(NUM_BS)]
            v_denominator = sum(interference)
            # print(i, bs, v_numerator, v_denominator, interference)
            V[bs] = np.clip(v_numerator / max(1e-35, v_denominator), 0.0, np.sqrt(power_max))
        V_map[:, i] = V

        "Update U"
        for bs in range(NUM_BS):
            ue_sched_g = bs * UE_PER_BS + sched_UEs[bs]
            u_numerator = np.sqrt(channel_map[ue_sched_g, bs]) * V[bs]
            interference = [channel_map[ue_sched_g, j] * V[j]**2 for j in range(NUM_BS)]
            u_denominator = sum(interference) + noise
            U[bs] = u_numerator / u_denominator

        "Update W"
        for bs in range(NUM_BS):
            ue_sched_g = bs * UE_PER_BS + sched_UEs[bs]
            E_i = 1.0 - U[bs] * np.sqrt(channel_map[ue_sched_g, bs]) * V[bs]
            W[bs] = 1.0 / max(1e-35, E_i)

    power = np.square(V)
    power_map = np.square(V_map)

    return power, power_map


class WMMSE_Power_Control:
    def __init__(self, bandwidth=None, method=None):
        self.bandwidth = bandwidth
        self.method = method

    def take_action(self, index=None, channels=None):

        throughput_profile = np.zeros(NUM_BS)
        power_profile = np.zeros(NUM_BS)

        sched_UEs = [0 for i in range(NUM_BS)]
        # channel_map, current_UEs = Channel_info.Select_UE_generate_Channel_map(seed=24)

        noise = noise_sigma(bandwidth=self.bandwidth)
        noise = dBm2Watt(noise)

        if index % 10 == 0:
            print('WMMSE: index {} | slot {}'.format(idx, SLOTS))

        "Choose the way to compute power"
        if self.method == 'WMMSE':
            power, power_map = WMMSE(num_iters=NUM_ITERS, channel_map=channels, power_max=P_MAX, noise=noise, weights=np.ones(NUM_BS))  # WMMSE
        elif self.method == 'FULL':
            power = P_MAX * np.ones(NUM_BS)  # Full Reused
        elif self.method == 'RANDOM':
            power = np.random.uniform(low=0, high=P_MAX, size=(NUM_BS,))  # Random

        power_profile = power

        "Compute throughput"
        interference, _ = compute_interference(channel_map=channel_map, power_profile=power_profile, num_BS=NUM_BS, num_UE_per_BS=UE_PER_BS)

        for bs in range(NUM_BS):
            ue_g = bs * UE_PER_BS + sched_UEs[bs]
            SINR = (power_profile[bs] * channel_map[bs][bs]) / (max(interference[bs], 1e-12) + noise)
            throughput_profile[bs] = alpha * np.log2(1 + SINR) - beta * power_profile[bs]

        return power_profile, throughput_profile  # slot profile [BS, ], [BS, ]
