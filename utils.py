import math
import numpy as np
import matplotlib.pyplot as plt
import time


def dBm2Watt(p):
    r"""convert dBm to Watt"""
    return 10**(p/10.0)/1000.0


def gen_init_power(n_BS, mode="zero", predef_power=None, tanh_factor=0.999):
    r"""generate initial power (under Tanh actor)
    Input:
        mode: "zero", "full", or "predefined", string type
        predef_power: if mode=="predefined", then need specify initial power
            size=(n_BS), entry range [0,1]
        tanh_factor: Tanh clip factor
    Output:
        acts: initial actions (not converted to real power yet), entry range [-factor,factor]
     """
    if mode == "zero":
        acts = [np.array([-tanh_factor]) for _ in range(n_BS)]  # tanh: -factor-> p=0
    elif mode == "full":
        acts = [np.array([tanh_factor]) for _ in range(n_BS)]  # factor -> p_max
    elif mode == "predefined":
        # acts = [np.array([predef_power[k]]) for k in range(n_BS)]
        acts = [np.array([unit_map_inverse(predef_power[k], factor=tanh_factor)]) for k in range(n_BS)]

    return acts


def compute_interference(channel_map, power_profile, num_BS, num_UE_per_BS):
    Interf_vec = np.zeros(num_BS)  # total interference at sched UEs
    Interf_mat = np.zeros([num_BS, num_BS])  # indiv. interf. of each BS to each sched UE
    Scheduled_UE = [0 for i in range(num_BS)]
    for bs in range(num_BS):
        ue_sched_g = bs * num_UE_per_BS + Scheduled_UE[bs]  # sched UE global index
        total_interf = 0.0  # total interference at BS bs's sched UE
        for bs_ in [x for x in range(num_BS) if x != bs]:
            temp = channel_map[ue_sched_g, bs_] * power_profile[bs_]
            total_interf += temp
        Interf_vec[bs] = total_interf
        """ individual interf from each BS bbss """
        for bbss in range(num_BS):
            ue_g = bs * num_UE_per_BS + Scheduled_UE[bs]  # sched UE of BS bs
            Interf_mat[bs, bbss] = channel_map[ue_g, bbss] * power_profile[bbss]
    return Interf_vec, Interf_mat

def unit_map(x, factor=1.0):
    """map [-factor,factor] to [0,1]"""
    return (np.array(x)+factor)/(2*factor)

def unit_map_inverse(x, factor=1.0):
    r"""map [0,1] to [-factor, factor]"""
    return (np.array(x) - 0.5) * 2*factor

def obs_list_to_state_vector(observation):
    """ Concat indiv obs into a single np array
    observation -> list of np arrays, contains local obs (1-D) of all agents:
    list([ array([x,y,z,...]), array([x',y',z'',...]),...])->array([x,y,z,...,x',y',z',...])
    This func only works for a single transition (no batch)
    """
    state = np.array([])  # 1D
    for obs in observation:
        state = np.concatenate([state, obs], axis=0)   # concat along dim 0 (default)
    return state

def obs_list_to_state_vector_2D(observation):
    """ Concat indiv obs into a single np array
    This func only works for a batched transitions (2D array)
    """
    state = observation[0]  #2D
    for i in range(1, len(observation)):
        state = np.concatenate([state, observation[i]], axis=1)
    return state

def moving_average(in_vec, window_size):
    """Compute moving average with fixed window size
        INPUT:
            in_vec: size = (|in_vec|, )
            window_size: scalar, average over past window_size slots
        OUTPUT:
            out_vec: size = (|in_sec|)
    """
    length = len(in_vec)
    in_vec = np.array(in_vec)    # convert to np array
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= window_size - 1:
            out_vec[i] = np.mean(in_vec[0: i+1])
        else:
            out_vec[i] = np.mean(in_vec[i - window_size + 1: i+1])
    return out_vec

# def plot_rwd_hist(Data):
#     r"""
#     :param: Data: input data, size(n_BS,n_slots)
#     :return: None
#     """
#     Data = np.array(Data)
#     n_BS = Data.shape[0]
#     fig, ax = plt.subplots(n_BS, sharex=True)
#     for i in range(n_BS):
#         ax[i].hist(Data[i], bins=50, edgecolor='black', label=f"User {i}")  # Adjust 'bins' as needed
#         ax[i].set_ylabel("Frequency")
#
#     ax[n_BS-1].set_xlabel("Reward (bits/Hz/sec)")
#     fig.suptitle("Reward distribution")
#     plt.savefig('reward_fig/fig_RwdHist_{}.pdf'.format(time.time()))
#     plt.show()


def noise_sigma(bandwidth=100 * 1e6):
    r""" Compute total noise over bandwidth
        INPUT:
            bwidth: bandwidth (Hz), default to 100 MHz
        OUTPUT:
            sigma: noise power
    """
    k_B = 1.38e-23  # Boltzman constant
    NR = 1.5  # noise figure (dB)
    T_0 = 290  # temperature (K)

    return 10 * math.log10(k_B * T_0 * 1000) + NR + 10 * math.log10(bandwidth)

def scale_data(data, max_val, min_val):
    """Normalize data to within range [0,1]"""
    return (data - min_val)/(max_val - min_val)

def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    print('pickle msg length: ', len(msg_pickle))
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.send(msg_pickle)
    print(msg[0], 'sent to', sock.getpeername())

def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    print('msg length: ', msg_len)
    # msg = sock.recv(msg_len)
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    print('length: ', len(msg))
    msg = pickle.loads(msg)
    print(msg[0], 'received from', sock.getpeername())

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg

def bandpass_filter(samps, fcenter, fpass, srate, order=5):
    """
    samps: The iq samples
    fcenter: The center frequency of your signal, which is shifted down to DC
    fpass: The bandwidth of your filter
    srate: The sample rate
    """
    nyq = 0.5*srate
    shift_seq = np.exp(-2j * np.pi * fcenter / srate * np.arange(len(samps)))  # multiple e^(-j*2*pi*fc*Ts*n) in time = frequency shift
    shift_samps = samps * shift_seq
    b, a = sig.butter(N=order, Wn=fpass/nyq, btype='low')  # b,a are polynomials
    fsamps = sig.lfilter(b, a, shift_samps)
    return fsamps / shift_seq

def mk_sine(nsamps, wampl, wfreq, srate):
    """

    Args:
        nsamps: Number of samples in the output signal
        wampl: Amplitude of the sinusoidal signal
        wfreq: Frequency of the sinusoidal signal
        srate: Sampling rate (number of samples per second)

    Returns: Complex signal

    """
    vals = np.ones((1, nsamps), dtype=np.complex64) * np.arange(nsamps)
    return wampl * np.exp(vals * 2j * np.pi * wfreq/srate)  # srate is the sample frequency
