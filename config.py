import argparse
from utils import *

parse = argparse.ArgumentParser()
parse.add_argument('-gamma', type=float, default=0.9, help='Discount factor')
parse.add_argument('-tau', type=float, default=0.001, help='tau value to compute Q value')
parse.add_argument('-bs', type=int, default=2, help='Number of Base Station')
parse.add_argument('-ue', type=int, default=2, help='Number of UEs')

# parse.add_argument('-ac_dim', type=int, default=5, help='Actor dimension (equal to number or BS)')
parse.add_argument('-p_max', type=float, default=15, help='Max Power in dBm (will be changed to power level in future)')
parse.add_argument('-num_ac', type=int, default=1, help='Number of actions for each agent (action dim)')

parse.add_argument('-slots', type=int, default=50000, help='Number of Slots')
parse.add_argument('-trails', type=int, default=3, help='Number of Trails (equal to number of seeds)')

parse.add_argument('-lr', type=list, default=[1e-4, 1e-3], help='Learning rate')
parse.add_argument('-noise_type', type=str, default='Gaussian', help='noise (for actor) type')
parse.add_argument('-noise_init', type=float, default=1.0, help='Initial actor noise value')
parse.add_argument('-noise_min', type=float, default=0.05, help='Minimum value of actor noise')
parse.add_argument('-noise_decay', type=float, default=1-7e-5, help='Noise decay value for actor noise')

parse.add_argument('-hidden_size', type=list, default=[200, 100, 32], help='list of hidden size for actor and critic network')
parse.add_argument('-buffer_size', type=int, default=500000, help='replay buffer size')
parse.add_argument('-batch_size', type=int, default=64, help='batch size for training')
parse.add_argument('-random_position', type=int, default=0, help='change the position during the MADDPG process or not (1 or 0)')

parse.add_argument('-power_mode', type=str, default='predefined', help='power mode for action initialization')
parse.add_argument('-act_function', type=str, default='tanh', help='activation function type like Tanh and Sigmoid')

parse.add_argument('-train_mode', type=bool, default=True, help='Train Mode type (true or false)')
parse.add_argument('-learn_freq', type=int, default=1, help='learning frequency (slots gap between learning)')
parse.add_argument('-gd_per_slot', type=int, default=1, help='gradient descent times per slot (local iterations)')

parse.add_argument('-num_iters', type=int, default=100, help='Number of iterations for WMMSE per slot')
parse.add_argument('-feedfreq', type=int, default=100, help='Number of slots for feedfreq')
parse.add_argument('-scaleinterf', type=bool, default=True, help='scale interference or not')

parse.add_argument('-server', type=str, default='127.0.0.1', help='server IP address')
parse.add_argument('-num_clients', type=int, default=2, help='number of clients connected to server')

args = parse.parse_args()

GAMMA = args.gamma
TAU = args.tau
NUM_BS = args.bs
NUM_UE = args.ue
UE_PER_BS = int(NUM_UE/NUM_BS)

ACTOR_DIM = 3 * NUM_BS + 5
ACTOR_DIMS = [ACTOR_DIM for _ in range(NUM_BS)]
CRITIC_DIM = ACTOR_DIM * NUM_BS
NUM_ACTIONS = args.num_ac

P_MAX_dBm = args.p_max
# P_MAX_dBm = 30
P_MAX = dBm2Watt(P_MAX_dBm)

SLOTS = args.slots
TRAILS = args.trails

ALPHA, BETA = args.lr  # Learning rate for target CNN and memory CNN
alpha = 1
beta = 0  # alpha and beta are using to compute the reward
NOISE_TYPE = args.noise_type
NOISE_INIT = args.noise_init
NOISE_MIN = args.noise_min
NOISE_DECAY = args.noise_decay

HIDDEN_SIZE = args.hidden_size
BUFFER_SIZE = args.buffer_size
BATCH_SIZE = args.batch_size

"Other Control Parameters"
LoadModel = False
POWER_MODE = args.power_mode
INIT_POWER = np.zeros(NUM_BS)
# FEEDFREQ = args.feedfreq
FEEDFREQ = SLOTS
ACT_FUNCTION = args.act_function
SHARE_REWARDS = True
SHUFFLE = args.random_position
ScaleInterf = args.scaleinterf

"Learning from memory parameters"
TRAIN_MODE = args.train_mode
LEARN_FREQ = args.learn_freq
GD_PER_SLOT = args.gd_per_slot

"Parameters for other baseline algorithm"
NUM_ITERS = args.num_iters
