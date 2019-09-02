import gym
import os
from Arguments import get_args
from RL_Agent_Models import DDPGAgent


###########################################################################
# Name: get_env_params
# Function: get the parameters of the environment provided by gym
# Comment:
###########################################################################
def get_env_params(env):
    obs = env.reset()
    dim_obs = obs['observation'].shape[0]
    dim_d_goal = obs['desired_goal'].shape[0]
    dim_action = env.action_space.shape[0]
    action_max = env.action_space.high[0]
    params = {'obs': dim_obs,
              'd_goal': dim_d_goal,
              'action': dim_action,
              'action_max': action_max,
              }
    params['max_timesteps'] = env._max_episode_steps

    return params


def launch(args):
    # create the environment
    env = gym.make(args.env_name)
    # get the environment parameters
    env_params = get_env_params(env)
    # create the DDPG agent to interact with the environment
    ddpg_trainer = DDPGAgent(args, env, env_params)
    # let the agent learn by itself
    ddpg_trainer.learning()


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    arguments = get_args()
    launch(arguments)
