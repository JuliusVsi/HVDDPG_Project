import matplotlib.pyplot as plt
import numpy as np


def smooth(data, weight=0.85):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed


if __name__ == '__main__':
    data_ddpg = np.loadtxt('plot_data/result_log_ddpg_1.txt', delimiter=',')
    x_ddpg = np.arange(len(data_ddpg))
    smoothed_data_ddpg = np.array(smooth(data_ddpg, weight=0.98))
    smoothed_data_ddpg_bar = np.array(smooth(data_ddpg, weight=0.9))
    np.savetxt('plot_data/smooth_ddpg.txt', smoothed_data_ddpg, fmt='%f', delimiter=',')
    d_data_ddpg = smoothed_data_ddpg - smoothed_data_ddpg_bar
    plt.plot(x_ddpg, smoothed_data_ddpg, label='DDPG')
    plt.fill_between(x_ddpg, smoothed_data_ddpg + d_data_ddpg, smoothed_data_ddpg-d_data_ddpg, color='blue',
                     alpha=0.1)


    data_vime = np.loadtxt('plot_data/result_log_ddpg_2.txt', delimiter=',')
    x_vime = np.arange(len(data_vime))
    smoothed_data_vime = np.array(smooth(data_vime, weight=0.98))
    smoothed_data_vime_bar = np.array(smooth(data_vime, weight=0.92))
    d_data_vime = smoothed_data_vime - smoothed_data_vime_bar
    np.savetxt('plot_data/smooth_vime.txt', smoothed_data_vime, fmt='%f', delimiter=',')

    plt.plot(x_vime, smoothed_data_vime, label='DDPG+VIME')
    plt.fill_between(x_vime, smoothed_data_vime + d_data_vime, smoothed_data_vime - d_data_vime, color='orange',
                     alpha=0.1)

    data_dqn = np.loadtxt('plot_data/result_DQN.txt', delimiter=',')
    x_dqn = np.arange(len(data_dqn))
    smoothed_data_dqn = np.array(smooth(data_dqn, weight=0.98))
    smoothed_data_dqn_bar = np.array(smooth(data_dqn, weight=0.9))
    d_data_dqn = smoothed_data_dqn - smoothed_data_dqn_bar
    np.savetxt('plot_data/smooth_dqn.txt', smoothed_data_dqn, fmt='%f', delimiter=',')

    plt.plot(x_dqn, smoothed_data_dqn, label='DQN')
    plt.fill_between(x_dqn, smoothed_data_dqn + d_data_dqn, smoothed_data_dqn - d_data_dqn, color='green',
                     alpha=0.1)

    plt.xlabel('Episodes')
    plt.ylabel('Reward of Each Episode')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
