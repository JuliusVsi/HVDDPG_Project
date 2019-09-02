import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())
total_steps = 0
all_rewards = []
for i in range(400):
    s_c = env.reset()
    done = False
    step = 0
    reward = 0

    while not done:
        env.render()
        action = natural_DQN.choose_action(s_c)

        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        s_n, r_n, done, info = env.step(np.array([f_action]))

        reward += r_n
        r_n = r_n / 10

        natural_DQN.store_transition(s_c, action, r_n, s_n)

        if total_steps > MEMORY_SIZE:   # learning
            natural_DQN.learn()

        step += 1
        total_steps += 1

        if step > (200-1):
            break

    all_rewards.append(reward)
    print('total steps: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f' % (total_steps, i, step, reward))

arr_reward = np.array(all_rewards)
np.savetxt('result_DQN.txt', arr_reward, fmt='%f', delimiter=',')
print(arr_reward)

