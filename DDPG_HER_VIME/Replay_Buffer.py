import numpy as np
import threading


###########################################################################
# Name: Replay_Buffer
# Function: Experience Replay for training the net
# Comment: Basically from the openAI baseline code
###########################################################################
class ReplayBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.max_timesteps = env_params['max_timesteps']
        self.size = buffer_size // self.max_timesteps
        self.sample_func = sample_func
        self.current_size = 0
        self.num_transitions_stored = 0

        # build the replay buffer to restore the transitions
        self.experience_buffer = {'obs': np.empty([self.size, self.max_timesteps + 1, self.env_params['obs']]),
                                  'a_goal': np.empty([self.size, self.max_timesteps + 1, self.env_params['d_goal']]),
                                  'd_goal': np.empty([self.size, self.max_timesteps, self.env_params['d_goal']]),
                                  'actions': np.empty([self.size, self.max_timesteps, self.env_params['action']]),
                                  }
        # using thread lock to control the resource
        self.lock = threading.Lock()

    # storing the transitions
    def store_transition(self, transition_batch):
        exp_obs, exp_a_goal, exp_d_goal, exp_actions = transition_batch
        batch_size = exp_obs.shape[0]
        with self.lock:
            index = self._get_store_index(increase_num=batch_size)
            # storing
            self.experience_buffer['obs'][index] = exp_obs
            self.experience_buffer['a_goal'][index] = exp_a_goal
            self.experience_buffer['d_goal'][index] = exp_d_goal
            self.experience_buffer['actions'][index] = exp_actions
            self.num_transitions_stored += self.max_timesteps * batch_size

    # sample the experience from the buffer
    def sample(self, sample_size):
        next_buffer = {}
        with self.lock:
            for key in self.experience_buffer.keys():
                next_buffer[key] = self.experience_buffer[key][:self.current_size]
        next_buffer['obs_next'] = next_buffer['obs'][:, 1:, :]
        next_buffer['a_goal_next'] = next_buffer['a_goal'][:, 1:, :]

        # sampling
        sampled_transitions = self.sample_func(next_buffer, sample_size)

        return sampled_transitions

    # get the index for storing the transitions
    def _get_store_index(self, increase_num=None):
        increase_num = increase_num or 1
        if self.current_size + increase_num <= self.size:
            index = np.arange(self.current_size, self.current_size + increase_num)
        elif self.current_size < self.size:
            beyond_num = increase_num - (self.size - self.current_size)
            index_temp_1 = np.arange(self.current_size, self.size)
            index_temp_2 = np.random.randint(0, self.current_size, beyond_num)
            index = np.concatenate([index_temp_1, index_temp_2])
        else:
            index = np.random.randint(0, self.size, increase_num)
        self.current_size = min(self.size, self.current_size + increase_num)

        if increase_num == 1:
            index = index[0]

        return index
