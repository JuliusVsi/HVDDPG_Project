import argparse


###########################################################################
# Name: get_args
# Function: get the arguments for the whole programme
# Comment:
###########################################################################
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env_name', type=str, default='FetchReach-v1', help='the environment name')
    parser.add_argument('--n_epochs', type=int, default=50, help='the number of epochs to train the agent')
    parser.add_argument('--n_cycles', type=int, default=50, help='the times to collect samples per epoch')
    parser.add_argument('--n_batches', type=int, default=40, help='the times to update the network')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay_strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip_return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise_epsilon', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random_epsilon', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer_size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay_ratio', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch_size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action_l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--learning_rate_actor', type=float, default=0.00001, help='the learning rate of the actor')
    parser.add_argument('--learning_rate_critic', type=float, default=0.00001, help='the learning rate of the critic')
    parser.add_argument('--avg_coeff', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n_eval', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo_length', type=int, default=5, help='the demo length')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--num_exp_per_mpi', type=int, default=2, help='the rollouts per mpi')

    args = parser.parse_args()

    return args
