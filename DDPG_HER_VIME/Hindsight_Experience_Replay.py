import numpy as np


###########################################################################
# Name: HER(Hindsight_Experience_Replay)
# Function: enrich the experience to deal with the sparse reward
# Comment:
###########################################################################
class HER:
    def __init__(self, strategy, replay_ratio, reward_func=None):
        self.strategy = strategy
        self.replay_ratio = replay_ratio
        self.reward_function = reward_func
        if self.strategy == 'future':
            self.future_num = 1 - (1. / (1 + replay_ratio))
        else:
            self.future_num = 0

    def her_sample_transitions(self, experience_batch, transitions_batch_size):
        timesteps = experience_batch['actions'].shape[1]
        exp_batch_size = experience_batch['actions'].shape[0]
        sample_batch_size = transitions_batch_size

        # select the experience to be used
        exp_index = np.random.randint(0, exp_batch_size, sample_batch_size)
        timesteps_sample = np.random.randint(timesteps, size=sample_batch_size)
        her_transitions = {key: experience_batch[key][exp_index, timesteps_sample].copy()
                           for key in experience_batch.keys()}

        # Index for HER
        her_index = np.where(np.random.uniform(size=sample_batch_size) < self.future_num)
        future_offset = np.random.uniform(size=sample_batch_size) * (timesteps - timesteps_sample)
        future_offset = future_offset.astype(int)
        future_timesteps = (timesteps_sample + 1 + future_offset)[her_index]

        # replace the desired goal in her transitions by the achieved goal
        future_a_goal = experience_batch['a_goal'][exp_index[her_index], future_timesteps]
        her_transitions['d_goal'][her_index] = future_a_goal

        # get the params to recalculate the rewards
        her_transitions['reward'] = np.expand_dims(self.reward_function(her_transitions['a_goal_next'],
                                                                        her_transitions['d_goal'], None), 1)
        her_transitions = {k: her_transitions[k].reshape(sample_batch_size, *her_transitions[k].shape[1:])
                           for k in her_transitions.keys()}

        return her_transitions
