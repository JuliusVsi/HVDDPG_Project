import threading
import numpy as np
from mpi4py import MPI


###########################################################################
# Name: MPI_Average
# Function: average across the cpu's data
# Comment:
###########################################################################
def mpi_average(x):
    buf = np.zeros_like(x)
    MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM)
    buf /= MPI.COMM_WORLD.Get_size()

    return buf


###########################################################################
# Name: Sync
# Function: sync the parameters across the cpus
# Comment:
###########################################################################
def sync(local_sum, local_sum_square, local_count):
    local_sum[...] = mpi_average(local_sum)
    local_sum_square[...] = mpi_average(local_sum_square)
    local_count[...] = mpi_average(local_count)

    return local_sum, local_sum_square, local_count


###########################################################################
# Name: Normalization
# Function: normalize the observation and goal
# Comment:
###########################################################################
class Normalizer:
    def __init__(self, size, epsilon=1e-2, clip_range=np.inf):
        self.size = size
        self.epsilon = epsilon
        self.clip_range = clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sum_square = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum, sum_square and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sum_square = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()

    # update the parameters of the normalizer
    def update(self, v):
        v = v.reshape(-1, self.size)
        # do the computing
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sum_square += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def recompute_stats(self):
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sum_square = self.local_sum_square.copy()
            # reset
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sum_square[...] = 0
        # sync the stats
        sync_sum, sync_sum_square, sync_count = sync(local_sum, local_sum_square, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sum_square += sync_sum_square
        self.total_count += sync_count
        # calculate the new mean and std
        self.mean = self.total_sum / self.total_count
        self.std = np.sqrt(np.maximum(np.square(self.epsilon), (self.total_sum_square / self.total_count) - np.square(
            self.total_sum / self.total_count)))

    # normalize the observation
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.clip_range
        return np.clip((v - self.mean) / self.std, -clip_range, clip_range)
