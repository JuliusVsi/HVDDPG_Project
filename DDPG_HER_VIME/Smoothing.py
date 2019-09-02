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
    ########################################################################
    # The Reach Figure
    ########################################################################
    plt.figure('Reach')
    reach_her = np.loadtxt('Plot_Data/Reach_HER.txt', delimiter=',')
    x_reach_her = np.arange(len(reach_her))
    smoothed_reach_her = np.array(smooth(reach_her, weight=0.4))
    smoothed_reach_her_bar = np.array(smooth(reach_her, weight=0.1))
    np.savetxt('Plot_Data/Smoothed_Reach_HER.txt', smoothed_reach_her, fmt='%f', delimiter=',')
    d_reach_her = smoothed_reach_her - smoothed_reach_her_bar
    plt.plot(x_reach_her, smoothed_reach_her, label='DDPG+HER')
    plt.fill_between(x_reach_her, smoothed_reach_her + d_reach_her, smoothed_reach_her - d_reach_her, color='blue',
                     alpha=0.1)

    reach_vime = np.loadtxt('Plot_Data/Reach_VIME.txt', delimiter=',')
    x_reach_vime = np.arange(len(reach_vime))
    smoothed_reach_vime = np.array(smooth(reach_vime, weight=0.4))
    smoothed_reach_vime_bar = np.array(smooth(reach_vime, weight=0.1))
    np.savetxt('Plot_Data/Smoothed_Reach_VIME.txt', smoothed_reach_vime, fmt='%f', delimiter=',')
    d_reach_vime = smoothed_reach_vime - smoothed_reach_vime_bar
    plt.plot(x_reach_vime, smoothed_reach_vime, label='HVDDPG')
    plt.fill_between(x_reach_vime, smoothed_reach_vime + d_reach_vime, smoothed_reach_vime - d_reach_vime,
                     color='orange', alpha=0.1)

    reach_ddpg = np.loadtxt('Plot_Data/Reach_DDPG.txt', delimiter=',')
    x_reach_ddpg = np.arange(len(reach_ddpg))
    smoothed_reach_ddpg = np.array(smooth(reach_ddpg, weight=0.6))
    smoothed_reach_ddpg_bar = np.array(smooth(reach_ddpg, weight=0.3))
    np.savetxt('Plot_Data/Smoothed_Reach_DDPG.txt', smoothed_reach_ddpg, fmt='%f', delimiter=',')
    d_reach_ddpg = smoothed_reach_ddpg - smoothed_reach_ddpg_bar
    plt.plot(x_reach_ddpg, smoothed_reach_ddpg, label='DDPG')
    plt.fill_between(x_reach_ddpg, smoothed_reach_ddpg + d_reach_ddpg, smoothed_reach_ddpg - d_reach_ddpg,
                     color='orange', alpha=0.1)

    reach_dqn = np.loadtxt('Plot_Data/Reach_DQN.txt', delimiter=',')
    x_reach_dqn = np.arange(len(reach_dqn))
    smoothed_reach_dqn = np.array(smooth(reach_dqn, weight=0.7))
    smoothed_reach_dqn_bar = np.array(smooth(reach_dqn, weight=0.0))
    np.savetxt('Plot_Data/Smoothed_Reach_DDPG.txt', smoothed_reach_dqn, fmt='%f', delimiter=',')
    d_reach_dqn = smoothed_reach_dqn - smoothed_reach_dqn_bar
    plt.plot(x_reach_dqn, smoothed_reach_dqn, label='DQN')
    plt.fill_between(x_reach_dqn, smoothed_reach_dqn + d_reach_dqn, smoothed_reach_dqn - d_reach_ddpg,
                     color='orange', alpha=0.1)

    plt.grid()
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend(loc='lower right')

    ########################################################################
    # The Push Figure
    ########################################################################
    plt.figure('Push')
    push_her = np.loadtxt('Plot_Data/Push_HER.txt', delimiter=',')
    x_push_her = np.arange(len(push_her))
    smoothed_push_her = np.array(smooth(push_her, weight=0.9))
    smoothed_push_her_bar = np.array(smooth(push_her, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Push_HER.txt', smoothed_push_her, fmt='%f', delimiter=',')
    d_push_her = smoothed_push_her - smoothed_push_her_bar
    temp_bar_1 = np.maximum(smoothed_push_her + d_push_her, 0)
    temp_bar_2 = np.maximum(smoothed_push_her - d_push_her, 0)

    plt.plot(x_push_her, smoothed_push_her, label='DDPG+HER')
    plt.fill_between(x_push_her, temp_bar_1, temp_bar_2, color='blue', alpha=0.1)

    push_vime = np.loadtxt('Plot_Data/Push_VIME.txt', delimiter=',')
    push_vime = push_vime[0:100]
    x_push_vime = np.arange(len(push_vime))
    smoothed_push_vime = np.array(smooth(push_vime, weight=0.9))
    smoothed_push_vime_bar = np.array(smooth(push_vime, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Push_VIME.txt', smoothed_push_vime, fmt='%f', delimiter=',')
    d_push_vime = smoothed_push_vime - smoothed_push_vime_bar
    temp_bar_1 = np.maximum(smoothed_push_vime + d_push_vime, 0)
    temp_bar_2 = np.maximum(smoothed_push_vime - d_push_vime, 0)

    plt.plot(x_push_vime, smoothed_push_vime, label='HVDDPG')
    plt.fill_between(x_push_vime, temp_bar_1, temp_bar_2, color='orange', alpha=0.1)

    push_ddpg = np.loadtxt('Plot_Data/Push_DDPG.txt', delimiter=',')
    x_push_ddpg = np.arange(len(push_ddpg))
    smoothed_push_ddpg = np.array(smooth(push_ddpg, weight=0.9))
    smoothed_push_ddpg_bar = np.array(smooth(push_ddpg, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Push_DDPG.txt', smoothed_push_ddpg, fmt='%f', delimiter=',')
    d_push_ddpg = smoothed_push_ddpg - smoothed_push_ddpg_bar
    temp_bar_1 = np.maximum(smoothed_push_ddpg + d_push_ddpg, 0)
    temp_bar_2 = np.maximum(smoothed_push_ddpg - d_push_ddpg, 0)

    plt.plot(x_push_ddpg, smoothed_push_ddpg, label='DDPG')
    plt.fill_between(x_push_ddpg, temp_bar_1, temp_bar_2, color='green', alpha=0.1)

    push_dqn = np.loadtxt('Plot_Data/Push_DQN.txt', delimiter=',')
    x_push_dqn = np.arange(len(push_dqn))
    smoothed_push_dqn = np.array(smooth(push_dqn, weight=0.9))
    smoothed_push_dqn_bar = np.array(smooth(push_dqn, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Push_DQN.txt', smoothed_push_dqn, fmt='%f', delimiter=',')
    d_push_dqn = smoothed_push_dqn - smoothed_push_dqn_bar
    temp_bar_1 = np.maximum(smoothed_push_dqn + d_push_dqn, 0)
    temp_bar_2 = np.maximum(smoothed_push_dqn - d_push_dqn, 0)

    plt.plot(x_push_dqn, smoothed_push_dqn, label='DQN')
    plt.fill_between(x_push_dqn, temp_bar_1, temp_bar_2, color='red', alpha=0.1)

    plt.grid()
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper left')

    ########################################################################
    # The Pick Figure
    ########################################################################
    plt.figure('Pick')
    pick_her = np.loadtxt('Plot_Data/Pick_HER.txt', delimiter=',')
    x_pick_her = np.arange(len(pick_her))
    smoothed_pick_her = np.array(smooth(pick_her, weight=0.98))
    smoothed_pick_her_bar = np.array(smooth(pick_her, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Pick_HER.txt', smoothed_pick_her, fmt='%f', delimiter=',')
    d_pick_her = smoothed_pick_her - smoothed_pick_her_bar
    temp_bar_1 = np.maximum(smoothed_pick_her + d_pick_her, 0)
    temp_bar_2 = np.maximum(smoothed_pick_her - d_pick_her, 0)

    plt.plot(x_pick_her, smoothed_pick_her, label='DDPG+HER')
    plt.fill_between(x_pick_her, temp_bar_1, temp_bar_2, color='blue', alpha=0.1)

    pick_vime = np.loadtxt('Plot_Data/Pick_VIME.txt', delimiter=',')
    x_pick_vime = np.arange(len(pick_vime))
    smoothed_pick_vime = np.array(smooth(pick_vime, weight=0.98))
    smoothed_pick_vime_bar = np.array(smooth(pick_vime, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Pick_VIME.txt', smoothed_pick_vime, fmt='%f', delimiter=',')
    d_pick_vime = smoothed_pick_vime - smoothed_pick_vime_bar
    temp_bar_1 = np.maximum(smoothed_pick_vime + d_pick_vime, 0)
    temp_bar_2 = np.maximum(smoothed_pick_vime - d_pick_vime, 0)

    plt.plot(x_pick_vime, smoothed_pick_vime, label='HVDDPG')
    plt.fill_between(x_pick_vime, temp_bar_1, temp_bar_2, color='orange', alpha=0.1)

    pick_ddpg = np.loadtxt('Plot_Data/Pick_DDPG.txt', delimiter=',')
    x_pick_ddpg = np.arange(len(pick_ddpg))
    smoothed_pick_ddpg = np.array(smooth(pick_ddpg, weight=0.9))
    smoothed_pick_ddpg_bar = np.array(smooth(pick_ddpg, weight=0.0))
    np.savetxt('Plot_Data/Smoothed_Pick_DDPG.txt', smoothed_pick_ddpg, fmt='%f', delimiter=',')
    d_pick_ddpg = smoothed_pick_ddpg - smoothed_pick_ddpg_bar
    temp_bar_1 = np.maximum(smoothed_pick_ddpg + d_pick_ddpg, 0)
    temp_bar_2 = np.maximum(smoothed_pick_ddpg - d_pick_ddpg, 0)

    plt.plot(x_pick_ddpg, smoothed_pick_ddpg, label='DDPG')
    plt.fill_between(x_pick_ddpg, temp_bar_1, temp_bar_2, color='green', alpha=0.1)

    pick_dqn = np.loadtxt('Plot_Data/Pick_DQN.txt', delimiter=',')
    x_pick_dqn = np.arange(len(pick_dqn))
    smoothed_pick_dqn = np.array(smooth(pick_dqn, weight=0.9))
    smoothed_pick_dqn_bar = np.array(smooth(pick_dqn, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Pick_DQN.txt', smoothed_pick_dqn, fmt='%f', delimiter=',')
    d_pick_dqn = smoothed_pick_dqn - smoothed_pick_dqn_bar
    temp_bar_1 = np.maximum(smoothed_pick_dqn + d_pick_dqn, 0)
    temp_bar_2 = np.maximum(smoothed_pick_dqn - d_pick_dqn, 0)

    plt.plot(x_pick_dqn, smoothed_pick_dqn, label='DQN')
    plt.fill_between(x_pick_dqn, temp_bar_1, temp_bar_2, color='red', alpha=0.1)

    plt.grid()
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper left')

    ########################################################################
    # The Slide Figure
    ########################################################################
    plt.figure('Slide')
    # her
    slide_her = np.loadtxt('Plot_Data/Slide_HER.txt', delimiter=',')
    x_slide_her = np.arange(len(slide_her))
    smoothed_slide_her = np.array(smooth(slide_her, weight=0.98))
    smoothed_slide_her_bar = np.array(smooth(slide_her, weight=0.8))
    np.savetxt('Plot_Data/Smoothed_Slide_HER.txt', smoothed_slide_her, fmt='%f', delimiter=',')
    d_slide_her = smoothed_slide_her - smoothed_slide_her_bar
    temp_bar_1 = np.maximum(smoothed_slide_her + d_slide_her, 0)
    temp_bar_2 = np.maximum(smoothed_slide_her - d_slide_her, 0)

    plt.plot(x_slide_her, smoothed_slide_her, label='DDPG+HER')
    plt.fill_between(x_slide_her, temp_bar_1, temp_bar_2, color='blue', alpha=0.1)

    # hvddpg
    slide_vime = np.loadtxt('Plot_Data/Slide_VIME.txt', delimiter=',')
    x_slide_vime = np.arange(len(slide_vime))
    smoothed_slide_vime = np.array(smooth(slide_vime, weight=0.98))
    smoothed_slide_vime_bar = np.array(smooth(slide_vime, weight=0.7))
    np.savetxt('Plot_Data/Smoothed_Slide_VIME.txt', smoothed_slide_vime, fmt='%f', delimiter=',')
    d_slide_vime = smoothed_slide_vime - smoothed_slide_vime_bar
    temp_bar_1 = np.maximum(smoothed_slide_vime + d_slide_vime, 0)
    temp_bar_2 = np.maximum(smoothed_slide_vime - d_slide_vime, 0)

    plt.plot(x_slide_vime, smoothed_slide_vime, label='HVDDPG')
    plt.fill_between(x_slide_vime, temp_bar_1, temp_bar_2, color='orange', alpha=0.1)

    # ddpg
    slide_ddpg = np.loadtxt('Plot_Data/Slide_DDPG.txt', delimiter=',')
    x_slide_ddpg = np.arange(len(slide_ddpg))
    smoothed_slide_ddpg = np.array(smooth(slide_ddpg, weight=0.98))
    smoothed_slide_ddpg_bar = np.array(smooth(slide_ddpg, weight=0.5))
    np.savetxt('Plot_Data/Smoothed_Slide_DDPG.txt', smoothed_slide_ddpg, fmt='%f', delimiter=',')
    d_slide_ddpg = smoothed_slide_ddpg - smoothed_slide_ddpg_bar
    temp_bar_1 = np.maximum(smoothed_slide_ddpg + d_slide_ddpg, 0)
    temp_bar_2 = np.maximum(smoothed_slide_ddpg - d_slide_ddpg, 0)

    plt.plot(x_slide_ddpg, smoothed_slide_ddpg, label='DDPG')
    plt.fill_between(x_slide_ddpg, temp_bar_1, temp_bar_2, color='green', alpha=0.1)

    slide_dqn = np.loadtxt('Plot_Data/Slide_DQN.txt', delimiter=',')
    x_slide_dqn = np.arange(len(slide_dqn))
    smoothed_slide_dqn = np.array(smooth(slide_dqn, weight=0.98))
    smoothed_slide_dqn_bar = np.array(smooth(slide_dqn, weight=0.0))
    np.savetxt('Plot_Data/Smoothed_Slide_DQN.txt', smoothed_slide_dqn, fmt='%f', delimiter=',')
    d_slide_dqn = smoothed_slide_dqn - smoothed_slide_dqn_bar
    temp_bar_1 = np.maximum(smoothed_slide_dqn + d_slide_dqn, 0)
    temp_bar_2 = np.maximum(smoothed_slide_dqn - d_slide_dqn, 0)

    plt.plot(x_slide_dqn, smoothed_slide_dqn, label='DQN')
    plt.fill_between(x_slide_dqn, temp_bar_1, temp_bar_2, color='red', alpha=0.1)

    plt.grid()
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper left')

    ########################################################################
    # The Egg Figure
    ########################################################################
    plt.figure('Egg')
    # her
    egg_her = np.loadtxt('Plot_Data/Egg_HER.txt', delimiter=',')
    x_egg_her = np.arange(len(egg_her))
    smoothed_egg_her = np.array(smooth(egg_her, weight=0.98))
    smoothed_egg_her_bar = np.array(smooth(egg_her, weight=0.3))
    np.savetxt('Plot_Data/Smoothed_Egg_HER.txt', smoothed_egg_her, fmt='%f', delimiter=',')
    d_egg_her = smoothed_egg_her - smoothed_egg_her_bar
    temp_bar_1 = np.maximum(smoothed_egg_her + d_egg_her, 0)
    temp_bar_2 = np.maximum(smoothed_egg_her - d_egg_her, 0)

    plt.plot(x_egg_her, smoothed_egg_her, label='DDPG+HER')
    plt.fill_between(x_egg_her, temp_bar_1, temp_bar_2, color='blue', alpha=0.1)

    # hvddpg
    egg_vime = np.loadtxt('Plot_Data/Egg_VIME.txt', delimiter=',')
    x_egg_vime = np.arange(len(egg_vime))
    smoothed_egg_vime = np.array(smooth(egg_vime, weight=0.98))
    smoothed_egg_vime_bar = np.array(smooth(egg_vime, weight=0.5))
    np.savetxt('Plot_Data/Smoothed_Egg_VIME.txt', smoothed_egg_vime, fmt='%f', delimiter=',')
    d_egg_vime = smoothed_egg_vime - smoothed_egg_vime_bar
    temp_bar_1 = np.maximum(smoothed_egg_vime + d_egg_vime, 0)
    temp_bar_2 = np.maximum(smoothed_egg_vime - d_egg_vime, 0)

    plt.plot(x_egg_vime, smoothed_egg_vime, label='HVDDPG')
    plt.fill_between(x_egg_vime, temp_bar_1, temp_bar_2, color='orange', alpha=0.1)

    # ddpg
    egg_ddpg = np.loadtxt('Plot_Data/Egg_DDPG.txt', delimiter=',')
    x_egg_ddpg = np.arange(len(egg_ddpg))
    smoothed_egg_ddpg = np.array(smooth(egg_ddpg, weight=0.98))
    smoothed_egg_ddpg_bar = np.array(smooth(egg_ddpg, weight=0.2))
    np.savetxt('Plot_Data/Smoothed_Egg_DDPG.txt', smoothed_egg_ddpg, fmt='%f', delimiter=',')
    d_egg_ddpg = smoothed_egg_ddpg - smoothed_egg_ddpg_bar
    temp_bar_1 = np.maximum(smoothed_egg_ddpg + d_egg_ddpg, 0)
    temp_bar_2 = np.maximum(smoothed_egg_ddpg - d_egg_ddpg, 0)

    plt.plot(x_egg_ddpg, smoothed_egg_ddpg, label='DDPG')
    plt.fill_between(x_egg_ddpg, temp_bar_1, temp_bar_2, color='green', alpha=0.1)


    ########################################################################
    # The pen Figure
    ########################################################################
    plt.figure('pen')

    # her
    pen_her = np.loadtxt('Plot_Data/Pen_HER.txt', delimiter=',')
    pen_her = pen_her[0:800]
    x_pen_her = np.arange(len(pen_her))
    smoothed_pen_her = np.array(smooth(pen_her, weight=0.99))
    smoothed_pen_her_bar = np.array(smooth(pen_her, weight=0.3))
    np.savetxt('Plot_Data/Smoothed_Pen_HER.txt', smoothed_pen_her, fmt='%f', delimiter=',')
    d_pen_her = smoothed_pen_her - smoothed_pen_her_bar
    temp_bar_1 = np.maximum(smoothed_pen_her + d_pen_her, 0)
    temp_bar_2 = np.maximum(smoothed_pen_her - d_pen_her, 0)

    plt.plot(x_pen_her, smoothed_pen_her, label='DDPG+HER')
    plt.fill_between(x_pen_her, temp_bar_1, temp_bar_2, color='blue', alpha=0.1)

    # hvddpg
    pen_vime = np.loadtxt('Plot_Data/Pen_VIME.txt', delimiter=',')
    x_pen_vime = np.arange(len(pen_vime))
    smoothed_pen_vime = np.array(smooth(pen_vime, weight=0.98))
    smoothed_pen_vime_bar = np.array(smooth(pen_vime, weight=0.5))
    np.savetxt('Plot_Data/Smoothed_Pen_VIME.txt', smoothed_pen_vime, fmt='%f', delimiter=',')
    d_pen_vime = smoothed_pen_vime - smoothed_pen_vime_bar
    temp_bar_1 = np.maximum(smoothed_pen_vime + d_pen_vime, 0)
    temp_bar_2 = np.maximum(smoothed_pen_vime - d_pen_vime, 0)

    plt.plot(x_pen_vime, smoothed_pen_vime, label='HVDDPG')
    plt.fill_between(x_pen_vime, temp_bar_1, temp_bar_2, color='orange', alpha=0.1)

    # ddpg
    pen_ddpg = np.loadtxt('Plot_Data/Pen_DDPG.txt', delimiter=',')
    x_pen_ddpg = np.arange(len(pen_ddpg))
    smoothed_pen_ddpg = np.array(smooth(pen_ddpg, weight=0.98))
    smoothed_pen_ddpg_bar = np.array(smooth(pen_ddpg, weight=0.5))
    np.savetxt('Plot_Data/Smoothed_Pen_DDPG.txt', smoothed_pen_ddpg, fmt='%f', delimiter=',')
    d_pen_ddpg = smoothed_pen_ddpg - smoothed_pen_ddpg_bar
    temp_bar_1 = np.maximum(smoothed_pen_ddpg + d_pen_ddpg, 0)
    temp_bar_2 = np.maximum(smoothed_pen_ddpg - d_pen_ddpg, 0)

    plt.plot(x_pen_ddpg, smoothed_pen_ddpg, label='DDPG')
    plt.fill_between(x_pen_ddpg, temp_bar_1, temp_bar_2, color='green', alpha=0.1)

    plt.grid()
    plt.ylim((-0.05, 1.05))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.legend(loc='upper left')
    plt.show()
