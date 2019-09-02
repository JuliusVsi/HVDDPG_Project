import numpy as np
import random
import pickle
import gym
import os


###############################################################
# Name: Soft Update
# Function: update target by target = tau * source + (1 - tau) * target
# Comment: param target: Target network, 
# param source: source network, 
# param tau: 0 < tau << 1
###############################################################
def soft_update(target, source, tau=0.001):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


###############################################################
# Name: Hard Update
# Function: update target by target = source
# Comment: param target: Target network, 
# param source: source network
###############################################################
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


###############################################################
# Name: Exploration Noise
# Function: create exploration noise 
# Comment: O-U process (Ornstein-Uhlenbeck process)
###############################################################
class OUNoise:
    def __init__(self, action_units, mu=0, theta=0.15, sigma=0.2):
        self.action_units = action_units
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_units) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_units) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


###############################################################
# Name: Replay Buffer
# Function: experience replay
# Comment: 
###############################################################
class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = []
        random.seed(random_seed)

    def add(self, s, a, r, t, s_n):
        experience = (s, a, r, t, s_n)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.pop(0)
            self.buffer.append(experience)
    
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s_n_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s_n_batch

    def clear(self):
        self.buffer = []
        self.count = 0

    def save(self):
        file = open('replay_buffer.obj', 'wb')
        pickle.dump(self.buffer, file)
        file.close()

    def load(self):
        try:
            file_handler = open('replay_buffer.obj', 'rb')
            self.buffer = pickle.load(file_handler)
            self.count = len(self.buffer)
        except:
            print('there was no file to load')


###############################################################
# Name: Get Output Folder
# Function: Return save folder
# Comment: 
###############################################################
def get_output_folder(parent_dir, env_name):
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


###############################################################
# Name: Environment Normalization
# Function: Normalize the action
# Comment: https://github.com/openai/gym/blob/master/gym/core.py
###############################################################
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.
        return act_k_inv * (action - act_b)


def load_obj(path):
    return pickle.load(open(path, 'rb'))
