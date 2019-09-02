import numpy as np
import matplotlib.pyplot as plt


ddpg_rewards = np.loadtxt('plot_data/result_log_ddpg_1.txt', delimiter=',')

vime_rewards = np.loadtxt('plot_data/result_log_ddpg_2.txt', delimiter=',')

x = np.arange(len(ddpg_rewards))
plt.plot(x, ddpg_rewards, x, vime_rewards)
plt.show()
np.savetxt('plot_data/ddpg_log.csv', ddpg_rewards, delimiter=',')
