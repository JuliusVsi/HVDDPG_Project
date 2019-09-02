import argparse
import os
import gym
from Configuration import Config
from Policy_Gradient import DDPG
from Train_Test import *
from Utilize import NormalizedEnv, load_obj


parser = argparse.ArgumentParser(description='')
parser.add_argument('--train', dest='train', action='store_true', help='train model')
parser.add_argument('--test', dest='test', action='store_true', help='test model')
parser.add_argument('--env', default='CartPole-v1', type=str, help='gym environment')
parser.add_argument('--gamma', default=0.99, type=float, help='discount')
parser.add_argument('--episodes', default=400, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epsilon', default=1.0, type=float, help='noise epsilon')
parser.add_argument('--eps_decay', default=0.01, type=float, help='epsilon decay')
parser.add_argument('--max_buff', default=1000000, type=int, help='replay buff size')
parser.add_argument('--output', default='out', type=str, help='result output dir')
parser.add_argument('--model_path', type=str, help='if test mode, import the model')
parser.add_argument('--load_config', type=str, help='load the config from obj file')

parser.add_argument('--max_steps', default=1000, type=int, help='max steps per episode')
'''
step_group = parser.add_argument_group('step')
step_group.add_argument('--customize_step', dest='customize_step', action='store_true', help='customize max step per episode')
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

'''
env = None
if args.customize_step:
    env = gym.make(config.env).env
else:
    env = gym.make(config.env)
'''
env = gym.make(config.env)
env = NormalizedEnv(env)
config.action_num = int(env.action_space.shape[0])
config.action_lim = float(env.action_space.high[0])
config.state_num = int(env.observation_space.shape[0])

if args.load_config is not None:
        config = load_obj(args.load_config)

agent = DDPG(config)

if args.train:
    trainer = Trainer(agent, env, config)
    trainer.train()

elif args.test:
    if args.model_path is None:
        print('please add the model path:', '--model_path xxxx')
        exit(0)

    tester = Tester(agent, env, model_path=args.model_path)
    tester.test()

else:
    print('choose train or test:', '--train or --test')
