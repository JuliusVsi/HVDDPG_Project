import numpy as np


class VimeReplayPool(object):
    def __init__(self, min_size=5000, max_size=50000, is_action_discrete=False):
        self.min_size = min_size
        self.max_size = max_size
        self.s_c, self.a_c, self.s_n = [], [], []
        self.current_size = 0
        self.is_action_discrete = is_action_discrete
        self.eps = 1e-8

    def fill_in(self, s_c, a_c, s_n):
        self.s_c = self.s_c + s_c
        self.a_c = self.a_c + a_c
        self.s_n = self.s_n + s_n
        # if the pool is full, then remove the early experience
        if len(self.s_c) > self.max_size:
            self.s_c = self.s_c[-self.max_size:]
            self.a_c = self.a_c[-self.max_size:]
            self.s_n = self.s_n[-self.max_size:]

        self.current_size = len(self.s_c)

    def get_mean_std(self):
        s_mean = np.mean(self.s_c, axis=0)
        s_std = np.std(self.s_c, axis=0)

        if self.is_action_discrete:
            a_mean = 0
            a_std = 1
        else:
            a_mean = np.mean(self.a_c, axis=0)
            a_std = np.std(self.a_c, axis=0)

        return s_mean, s_std, a_mean, a_std

    ##########################
    # Sample Data
    # Function: Sample the data for training
    # Comment: normalize refers to normalize the input for BNN
    ##########################
    def sample_data(self, sample_pool_size=5000, normalize=True):
        assert self.current_size > self.min_size      # check if sufficient data to be sampled
        if self.current_size >= sample_pool_size:
            index = np.random.randint(self.current_size, size=sample_pool_size)
            s_c_sample = np.take(np.array(self.s_c), index, axis=0)
            a_c_sample = np.take(np.array(self.a_c), index, axis=0)
            s_n_sample = np.take(np.array(self.s_n), index, axis=0)
        else:
            s_c_sample = np.array(self.s_c[-sample_pool_size:])
            a_c_sample = np.array(self.a_c[-sample_pool_size:])
            s_n_sample = np.array(self.s_n[-sample_pool_size:])

        # normalize input if required
        if normalize:
            s_mean, s_std, a_mean, a_std = self.get_mean_std()

            s_c_sample = (s_c_sample - s_mean) / (s_std + self.eps)
            a_c_sample = (a_c_sample - a_mean) / (a_std + self.eps)
            s_n_sample = (s_n_sample - s_mean) / (s_std + self.eps)

            print('Current observation mean/std is %s +/-%s' % (str(s_mean), str(s_std)))
            print('Current action mean/std is %s +/-%s' % (str(a_mean), str(a_std)))
        bnn_input = np.column_stack((s_c_sample, a_c_sample))
        bnn_output = s_n_sample

        return bnn_input, bnn_output
