import random
import numpy as np
import os
import time
from config import *
from maddpg import *
from utils import *
import matplotlib.pyplot as plt
from WMMSE import WMMSE

class MADDPG_Power_Control:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.score_history = []

    def take_action(self, index=None, channels=None):

        channel_map, next_channel_map = channels  # channels = [current, next]  channel_map should be a matrix with [BS, BS] dimension
        start_time = time.time()
        noise = noise_sigma(bandwidth=self.bandwidth)
        noise = dBm2Watt(noise)

        "Apply if scalar data needed"
        channel_max = np.max(channel_map, axis=0)
        Interf_max, _ = compute_interference(channel_map=channel_max, power_profile=P_MAX * np.ones(NUM_BS),
                                             num_BS=NUM_BS, num_UE_per_BS=UE_PER_BS)
        Interf_max += noise

        MADDPG_agents = MADDPG(actor_dims=ACTOR_DIMS, critic_dims=CRITIC_DIM, n_agents=NUM_BS, n_actions=NUM_ACTIONS,
                               fc1=HIDDEN_SIZE[0], fc2=HIDDEN_SIZE[1], fc3=HIDDEN_SIZE[2], alpha=ALPHA, beta=BETA,
                               gamma=GAMMA, tau=TAU, action_noise=NOISE_TYPE, noise_init=NOISE_INIT, noise_min=NOISE_MIN,
                               noise_decay=NOISE_DECAY, chkpt_dir="./tmp/maddpg/")

        if LoadModel and len(os.listdir("./tmp/maddpg")) > 1:
            MADDPG_agents.load_checkpoint()

        memory = MultiAgentReplayBuffer(max_size=BUFFER_SIZE, critic_dims=CRITIC_DIM, actor_dims=ACTOR_DIMS,
                                        n_actions=NUM_ACTIONS, n_agents=NUM_BS, batch_size=BATCH_SIZE)

        interf_profile = np.zeros(NUM_BS, dtype=np.float32)
        interf_profile_ = np.zeros(NUM_BS, dtype=np.float32)  # decision point interf
        indiv_interf_profile = np.zeros([NUM_BS, NUM_BS], dtype=np.float32)
        indiv_interf_profile_ = np.zeros([NUM_BS, NUM_BS], dtype=np.float32)  # DP interf g(t+1)p(t)
        reward_profile = np.zeros(NUM_BS, dtype=np.float32)
        power_profile = np.zeros(NUM_BS, dtype=np.float32)
        factor = 0.95

        obs_list_ = [np.zeros(ACTOR_DIM) for _ in range(NUM_BS)]
        state_ = np.zeros(CRITIC_DIM)

        "Make the Power decision using RL algorithm"
        if index == 0:
            obs_list = [np.zeros(ACTOR_DIM) for _ in range(NUM_BS)]
            state = state_
            actions = gen_init_power(n_BS=NUM_BS, mode=POWER_MODE, predef_power=np.zeros(NUM_BS),
                                     tanh_factor=factor)
        else:
            obs_list = obs_list_
            state = state_
            if index % FEEDFREQ == 0:
                power_wmmse, _ = WMMSE(num_iters=2000, channel_map=channel_map, power_max=P_MAX,
                                       noise=noise, weights=np.ones(NUM_BS))
                actions = gen_init_power(n_BS=NUM_BS, mode='predefined', predef_power=power_wmmse/P_MAX,
                                         tanh_factor=factor)
            else:
                actions = MADDPG_agents.choose_action(np.array(obs_list, dtype=np.float32))
        if ACT_FUNCTION == 'tanh':
            power_profile = np.abs(np.array([P_MAX*(actions[i].item()+factor)/(2*factor) for i in range(NUM_BS)]))
        elif ACT_FUNCTION == 'sigmoid':
            power_profile = np.array([P_MAX * actions[i].item() for i in range(NUM_BS)])
        else:
            raise Exception('The activation function is not embedded')

        "Compute interference (g_ij*p_j) for current slot"
        inter_vec, inter_mat = compute_interference(channel_map=channel_map, power_profile=power_profile,
                                                    num_BS=NUM_BS, num_UE_per_BS=UE_PER_BS)
        interf_profile = inter_vec
        indiv_interf_profile = inter_mat

        "Compute interference (g_ij*p_j) for next slot (decision point), channel_map will be changed in future"
        inter_vec_, inter_mat_ = compute_interference(channel_map=next_channel_map,
                                                      power_profile=power_profile, num_BS=NUM_BS,
                                                      num_UE_per_BS=UE_PER_BS)
        interf_profile_ = inter_vec_
        indiv_interf_profile_ = inter_mat_

        "Compute reward (total capacity C (throughput/sec/Hz) for each BS)"
        for bs in range(NUM_BS):
            SINR = (power_profile[bs] * channel_map[bs][bs]) / (max(interf_profile[bs], 1e-12) + noise)
            reward_profile[bs] = alpha * np.log2(1 + SINR) - beta * power_profile[bs]
        # print(idx, reward_profile[:, idx], '\n')
        self.score_history.append(np.mean(reward_profile))
        "next state (local observation)"
        obs_list_ = []
        for i in range(NUM_BS):
            local_obs = []
            local_obs.append(power_profile[i])  # p_i^t
            local_obs.append(reward_profile[i] / max(sum(reward_profile), 1e-20))  # C_i/sum(C_j)

            local_obs.append(sum(reward_profile))  # locally shared rwd grp? What is this parameter???
            local_obs += list(reward_profile)  # C_j^t

            local_obs.append(channel_map[i][i])  # g_ii^t
            local_obs.append(next_channel_map[i][i])  # g_ii^(t+1) will be changed when apply to the real condition

            if ScaleInterf:
                local_obs.append(scale_data(interf_profile[i] + noise, Interf_max[i], 0.0))  # I_i^t
                local_obs.append(scale_data(interf_profile_[i] + noise, Interf_max[i], 0.0))  # I_i^(t+1)

                local_obs += [scale_data(indiv_interf_profile[i, j] + noise, Interf_max[i], 0.0)
                              for j in range(NUM_BS) if j != i]  # g_ij^t*p_j^t
                local_obs += [scale_data(indiv_interf_profile_[i, j] + noise, Interf_max[i], 0.0)
                              for j in range(NUM_BS) if j != i]  # g_ij^(t+1)*p_j^t
            else:
                local_obs.append(interf_profile[i] + noise)  # I_i^t
                local_obs.append(interf_profile_[i] + noise)  # I_i^(t+1)

                local_obs += [indiv_interf_profile[i, j] + noise for j in range(NUM_BS) if j != i]  # g_ij^t*p_j^t
                local_obs += [indiv_interf_profile_[i, j] + noise for j in range(NUM_BS) if j != i]  # g_ij^(t+1)*p_j^t

            obs_list_.append(local_obs)
        state_ = obs_list_to_state_vector(obs_list_)
        index += 1

        "Store the experience"
        done = False
        rwd_list = [sum(reward_profile) for _ in range(NUM_BS)] if SHARE_REWARDS else [reward_profile[i] for i in range(NUM_BS)]
        memory.store_transition(obs_list, state, actions, rwd_list, obs_list_, state_, done)

        "Learn with a minibatch"
        if TRAIN_MODE:
            if index % LEARN_FREQ == 0:
                for _ in range(GD_PER_SLOT):
                    MADDPG_agents.learn(memory)  # Learning from memory
        else:
            pass

        "Print progress"
        if index % 20 == 0:
            print('---------------------------------------------')
            print('SLOT: {} | REWARD: {} | TRAIL_AVG_RWD: {}'.format(index, self.score_history[index], np.mean(self.score_history[-50:])))

            if ACT_FUNCTION == 'tanh':
                print(f">Acts(normed):{'%.2f|' * NUM_BS}" % tuple(np.abs(
                    unit_map([actions[i].item() for i in range(NUM_BS)], factor=factor))))
            else:
                print(f">Acts (normed):{'%.2f|' * NUM_BS}" % tuple([actions[i].item() for i in range(n_BS)]))
            if NOISE_TYPE == "Gaussian":
                print(f">Expl_noise_std:{MADDPG_agents.agents[0].noise.std:.4f}")
            else:
                print(f">Expl_noise_width:{MADDPG_agents.agents[0].noise.width:.4f}")

            print('---------------------------------------------')

        "save checkpoint"
        MADDPG_agents.save_checkpoint()
        # average_throughput = np.mean(reward_profile)

        return power_profile, reward_profile  # slot profile [BS, ], [BS, ]
