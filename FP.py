import numpy as np
import os
import time
from config import *
from utils import *
import matplotlib.pyplot as plt


def Fractional_Programming(channel_map, power_max, noise, num_iters, init, weights, p_init_factor=0.5, if_gamma_auto=False, gamma_init=0.1):
    """FP with properly-initialized gammas, where gamma is SINR"""

    UE_sched = [0 for i in range(NUM_BS)]
    channel_map = np.array(channel_map)
    Gamma_profile = np.zeros([NUM_BS, num_iters])
    Power_profile = np.zeros([NUM_BS, num_iters])

    "Init P, gamma, Y"
    c = np.random.uniform(0.1, 0.9) if init == 'random' else p_init_factor
    P = c * power_max * np.ones(NUM_BS)

    SINR_max = np.zeros(NUM_BS)  # Max possible value for each base station
    for bs in range(NUM_BS):
        SINR_max[bs] = channel_map[bs * UE_PER_BS + UE_sched[bs], bs] * power_max / noise
    gamma = SINR_max if if_gamma_auto else gamma_init * np.ones(NUM_BS)
    Y = np.zeros(NUM_BS)

    "begin iteration"
    for idx in range(num_iters):
        "Update Y"
        for i in range(NUM_BS):
            ue_g = i * UE_PER_BS + UE_sched[i]
            numerator = np.sqrt(weights[i] * (1.0 + gamma[i]) * channel_map[ue_g, i] * P[i])
            interference = [channel_map[ue_g, j] * P[j] for j in range(NUM_BS)]
            denominator = sum(interference) + noise
            Y[i] = numerator / max(denominator, 1e-50)

        "Update gamma (SINR)"
        for i in range(NUM_BS):
            ue_g = i * UE_PER_BS + UE_sched[i]
            numerator = channel_map[ue_g, i] * P[i]
            interference = [channel_map[ue_g, j] * P[j] for j in range(NUM_BS) if j != i]
            denominator = sum(interference) + noise
            gamma[i] = numerator / max(denominator, 1e-50)
        Gamma_profile[:, idx] = gamma

        "Update P"
        for i in range(NUM_BS):
            ue_g = i * UE_PER_BS + UE_sched[i]
            numerator = Y[i]**2 * weights[i] * (1.0 + gamma[i]) * channel_map[ue_g, i]
            interference = np.zeros(NUM_BS)
            for j in range(NUM_BS):
                interference[j] = Y[j]**2 * channel_map[j * UE_PER_BS + UE_sched[j], i]
            denominator = np.sum(interference)**2
            P[i] = min(power_max, numerator / max(denominator, 1e-50))

        Power_profile[:, idx] = P

    return P, Power_profile, Gamma_profile, SINR_max


class FP_Power_Control:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def take_action(self, index=None, channels=None):

        noise = noise_sigma(bandwidth=220e3)
        noise = dBm2Watt(noise)
        num_iters = NUM_ITERS

        UE_sched = [0 for i in range(NUM_BS)]

        if index % 50 == 0:
            print('FP: slot {}'.format(index))

        "Compute FP Powers"
        power, power_map, gamma_map, _ = Fractional_Programming(channel_map=channels, power_max=P_MAX,
                                                                noise=noise, num_iters=num_iters,
                                                                init='constant', weights=np.ones(NUM_BS))

        "Compute throughput/FP objective"
        weights = np.ones(NUM_BS)
        obj_val = np.zeros(num_iters)
        throughput = np.zeros(num_iters)

        for iter_idx in range(num_iters):
            array1 = np.array([weights[i] * np.log2(1 + gamma_map[i, iter_idx]) for i in range(NUM_BS)])
            array2 = np.array([weights[i] * gamma_map[i, iter_idx] for i in range(NUM_BS)])
            array_numerator = np.array([weights[i] * (1 + gamma_map[i, iter_idx]) * channel_map[i * UE_PER_BS + UE_sched[i], i] * power_map[i, iter_idx] for i in range(NUM_BS)])
            array_denomiator = np.zeros(NUM_BS)
            for i in range(NUM_BS):
                array_denomiator[i] = sum([channel_map[i * UE_PER_BS + UE_sched[i], j] * power_map[j, iter_idx] for j in range(NUM_BS)]) + noise
            array3 = np.array([array_numerator[i] / array_denomiator[i] for i in range(NUM_BS)])

            value1 = sum(array1)
            value2 = sum(array2)
            value3 = sum(array3)

            throughput[iter_idx] = value1 / NUM_BS
            obj_val[iter_idx] = (value1 - value2 + value3) / NUM_BS

        # Average_throughput = np.mean(throughput)

        return power, throughput  # slot feedback [BS, ], 1


