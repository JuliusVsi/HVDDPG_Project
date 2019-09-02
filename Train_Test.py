import numpy as np
from Configuration import Config
from Utilize import get_output_folder


###############################################################
# Name: Trainer
# Function: train the model
# Comment: 
###############################################################
class Trainer(object):
    def __init__(self, agent, env, config: Config):
        self.agent = agent
        self.config = config
        self.outputdir = get_output_folder(self.config.output, self.config.env)
        
        self.env = env
        self.env.seed(config.seed)
        
    def train(self, pre_episodes=0, pre_total_step=0):
        total_step = pre_total_step
        all_rewards = []

        for ep in range(pre_episodes + 1, self.config.episodes + 1):
            s_c = self.env.reset()
            self.agent.reset()

            done = False
            step = 0
            actor_loss, critic_loss, reward = 0, 0, 0

            # decay noise
            self.agent.decay_epsilon()

            while not done:
                if self.config.RENDER:
                    self.env.render()
                
                action = self.agent.get_action(s_c)
                s_n, r_n, done, info = self.env.step(action)
                self.agent.buffer.add(s_c, action, r_n, done, s_n)
                s_c = s_n

                if self.agent.buffer.size() > self.config.batch_size:
                    loss_a, loss_c = self.agent.learning()
                    actor_loss += loss_a
                    critic_loss += loss_c

                reward += r_n
                step += 1
                total_step += 1

                if step > (self.config.max_steps - 1):
                    break

            all_rewards.append(reward)
            # avg_reward = np.mean(all_rewards[-20:])

            print('total step: %5d, episodes %3d, episode_step: %5d, episode_reward: %5f'
                  % (total_step, ep, step, reward))

        self.env.close()
        # save model at last
        self.agent.save_model(self.outputdir)
        # print('avg_reward', avg_reward)
        arr_reward = np.array(all_rewards)
        np.savetxt('plot_data/result_log_ddpg_2.txt', arr_reward, fmt='%f', delimiter=',')
        print(arr_reward)

###############################################################
# Name: Tester
# Function: test the trained model
# Comment: 
###############################################################
class Tester(object):
    def __init__(self, agent, env, model_path, num_episodes=50, test_ep_steps=400):
        self.num_episodes = num_episodes
        self.test_ep_steps = test_ep_steps
        self.agent = agent

        self.env = env
        self.agent.load_weights(model_path)

    def test(self, debug=False):
        avg_reward = 0
        for episode in range(self.num_episodes):

            # reset at the start of episode
            s_c = self.env.reset()
            episode_steps = 0
            episode_reward = 0.

            # start episode
            done = False
            while not done:

                self.env.render()

                action = self.agent.get_action(s_c)

                s_c, reward, done, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                if episode_steps + 1 > self.test_ep_steps:
                    done = True

            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))

            avg_reward += episode_reward

        avg_reward /= self.num_episodes

        print("avg reward: %5f" % avg_reward)
