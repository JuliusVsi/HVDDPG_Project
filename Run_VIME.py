import argparse
import gym
import copy
import torch
from Configuration import Config
from Policy_Gradient import DDPG
from BNN import *
from Train_Test import *
from BNN_Train import BNNTrainer
from Utilize import NormalizedEnv
from Vime_Buffer import VimeReplayPool
from Vime_Util import build_test_set, update_kl_normalization
from collections import deque


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train', action='store_true', help='train model')
parser.add_argument('--test', dest='test', action='store_true', help='test model')
parser.add_argument('--env', default='Pendulum-v0', type=str, help='gym environment')
parser.add_argument('--gamma', default=0.99, type=float, help='discount')
parser.add_argument('--episodes', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epsilon', default=1.0, type=float, help='noise epsilon')
parser.add_argument('--eps_decay', default=0.001, type=float, help='epsilon decay')
parser.add_argument('--max_buff', default=1000000, type=int, help='replay buff size')
parser.add_argument('--output', default='out', type=str, help='result output dir')
parser.add_argument('--model_path', type=str, help='if test mode, import the model')
parser.add_argument('--load_config', type=str, help='load the config from obj file')

parser.add_argument('--max_steps', default=1000, type=int, help='max steps per episode')
'''
step_group = parser.add_argument_group('step')
step_group.add_argument('--customize_step', dest='customize_step', action='store_true', \
help='customize max step per episode')
step_group.add_argument('--max_steps', default=1000, type=int, help='max steps per episode')
'''

args = parser.parse_args()
config = Config()
config.env = args.env
config.gamma = args.gamma
config.episodes = args.episodes
config.max_steps = args.max_steps
config.batch_size = args.batch_size
config.epsilon = args.epsilon
config.eps_decay = args.eps_decay
config.max_buff_size = args.max_buff
config.output = args.output

config.learning_rate_actor = 1e-4
config.learning_rate_critic = 1e-3
config.epsilon_min = 0.001
config.epsilon = 1.0
config.tau = 0.001

env = gym.make(config.env)
env = NormalizedEnv(env)

config.action_num = int(env.action_space.shape[0])
config.action_lim = float(env.action_space.high[0])
config.state_num = int(env.observation_space.shape[0])

###############################################################
# Hyper-parameters for VIME
###############################################################
lr_bnn = 0.001
batch_size_bnn = 10
max_epoch_bnn = 5
extra_w_kl = 0.1
input_units = config.state_num + 1
hidden_units = 32
output_units = config.state_num
prior_std_bnn = 0.5
likelihood_ds_bnn = 1.0
sample_pool_size = 5000
stop_train_vime = False

# previous kl used for normalization
kl_q_l = 100
# whether normalize the intrinsic reward
use_kl_q = True
# weights of intrinsic reward
eta = 0.25

log_cfg = {
    'display_interval': 10000,
    'val_interval': max_epoch_bnn,
}

extra_arg = {
    'external_criterion_val': torch.nn.MSELoss(),
}

# initialize the DDPG class
agent = DDPG(config)

# initialize the BNN class
# Comment: inputs_num = state_num + 1;
# outputs_num = state_num
bnn = BayesianNeuralNetwork(input_units, hidden_units, output_units,
                            prior_std=prior_std_bnn, likelihood_sd=likelihood_ds_bnn)

optimizer_bnn = torch.optim.Adam(bnn.parameters(), lr=lr_bnn)

# initialize the replay pool for vime
replay_pool = VimeReplayPool(min_size=1000, max_size=10000, is_action_discrete=False)

# ??????

train_set = None
test_set = None

kl_normalize = 1
pre_kl_medians = deque(maxlen=kl_q_l)

all_rewards = []

for i_episode in range(config.episodes):
    print('-----------------Episode %d---start-----------------' % i_episode)
    s_c = env.reset()
    agent.reset()
    episode_step = 0
    episode_reward = 0
    # decay noise
    agent.decay_epsilon()

    while True:
        if config.RENDER:
            env.render()

        action = agent.get_action(s_c)
        episode_step += 1
        s_n, reward, done, info = env.step(action)

        episode_reward += reward
        agent.ep_next_obs.append(s_n)
        agent.ep_naive_rewards.append(reward)
        # ???
        naive_all, intrinsic_all = [], []
        naive_all.append(reward)
        # calculate the kl
        if replay_pool.current_size > replay_pool.min_size:
            test_x, test_y = build_test_set(s_c, action, s_n, replay_pool, is_norm=True)  # add intrinsic reward
            intrinsic_reward = bnn.kl_second_order_approx(step_size=1e-2, inputs=test_x, targets=test_y).item()
        else:
            intrinsic_reward = 0
        agent.ep_kls.append(copy.deepcopy(intrinsic_reward))      # deep copy ??????
        if use_kl_q and kl_normalize != 0:
            intrinsic_reward /= kl_normalize
        reward += eta * intrinsic_reward

        agent.buffer.add(s_c, action, reward, done, s_n)
        agent.store_transition(s_c, action, reward)
        s_c = s_n

        # train the general policy
        if agent.buffer.size() > config.batch_size:
            loss_a, loss_c = agent.learning()

        if done or (episode_step > config.max_steps - 1):
            break
    all_rewards.append(episode_reward)
    print('episode_step: %5d, episode_reward: %5f' % (episode_step, episode_reward))

    replay_pool.fill_in(agent.ep_obs, agent.ep_actions, agent.ep_next_obs)
    if use_kl_q:
        kl_normalize = update_kl_normalization(agent.ep_kls, pre_kl_medians)
    if stop_train_vime == False :
        if replay_pool.current_size > replay_pool.min_size:
            train_set = replay_pool.sample_data(sample_pool_size=sample_pool_size, normalize=True)

            # train the bayesian neural network
            bnn_trainer = BNNTrainer(bnn, train_set, batch_size_bnn,
                                     optimizer_bnn, max_epoch_bnn, log_cfg,
                                     extra_w_kl, test_set, extra_arg)
            loss_bnn, metric_train, kl_train = bnn_trainer.train()
            print('loss_bnn: %5f, metric_train: %5f, kl_train: %5f'
                  % (loss_bnn, metric_train, kl_train))
            if metric_train < 0.005:
                # stop_train_vime = True
                print('Stop Training VIME!!!!!')
    print('-----------------Episode %d---end-----------------' % i_episode)
env.close()

arr_reward = np.array(all_rewards)
np.savetxt('result_log_vime.txt', arr_reward, fmt='%f', delimiter=',')

# result replay
print('Starting Result Play......')

test_times = 0
max_test_eps_steps = 500
all_test_reward = []
while True:
    obs_test = env.reset()
    test_step = 0
    test_eps_reward = 0

    test_done = False

    while not test_done:
        env.render()

        test_action = agent.get_action(obs_test)
        test_step += 1
        obs_next_test, reward_test, test_done, test_info = env.step(test_action)

        test_eps_reward += reward_test
        test_step += 1

        if test_step + 1 > max_test_eps_steps:
            test_done = True

    print('test_times = %f  eps_steps = %f  eps_reward = %f  ' % (test_times, test_step, test_eps_reward))
    test_times += 1
