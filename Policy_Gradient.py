import torch.nn.functional as fun
import torch.optim as optim
from Net_Module import *
from Configuration import Config
from Utilize import soft_update, hard_update, OUNoise, ReplayBuffer


class DDPG(object):
    def __init__(self, config: Config):
        self.config = config

        self.actions_num = self.config.action_num
        self.states_num = self.config.state_num
        self.batch_size = self.config.batch_size
        self.max_buff_size = self.config.max_buff_size
        self.tau = self.config.tau
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.eps_decay = self.config.eps_decay
        self.epsilon_min = self.config.epsilon_min
        # self.reward_decay = reward_decay
        self.randomer = OUNoise(self.actions_num)
        self.buffer = ReplayBuffer(self.max_buff_size)

        # observations, actions, and rewards
        self.ep_obs, self.ep_actions, self.ep_rewards = [], [], []

        # add additional to store next observation, and original reward
        self.ep_next_obs, self.ep_naive_rewards, self.ep_kls = [], [], []

        # _build_net
        self.actor = Actor(self.states_num, self.actions_num)
        self.actor_target = Actor(self.states_num, self.actions_num)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.config.learning_rate_actor)

        self.critic = Critic(self.states_num, self.actions_num)
        self.critic_target = Critic(self.states_num, self.actions_num)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.config.learning_rate_critic)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    # Decay Epsilon
    def decay_epsilon(self):
        self.epsilon -= self.eps_decay

    def learning(self):
        s_c, a_c, r_c, result, s_n = self.buffer.sample_batch(self.batch_size)
        result = (result == False) * 1     # bool -> int
        s_c = torch.tensor(s_c, dtype=torch.float)
        a_c = torch.tensor(a_c, dtype=torch.float)
        r_c = torch.tensor(r_c, dtype=torch.float)
        result = torch.tensor(result, dtype=torch.float)
        s_n = torch.tensor(s_n, dtype=torch.float)

        a_n = self.actor_target.forward(s_n).detach()
        target_q = self.critic_target.forward(s_n, a_n).detach()
        y_expected = r_c[:, None] + result[:, None] * self.gamma * target_q
        y_predicted = self.critic.forward(s_c, a_c)

        # Critic Gradient
        critic_loss = torch.nn.MSELoss()
        loss_critic = critic_loss(y_predicted, y_expected)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # Actor Gradient
        a_predicted = self.actor.forward(s_c)
        loss_actor = (-self.critic.forward(s_c, a_predicted)).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # update the target net using soft replacement
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
        
        return loss_actor.item(), loss_critic.item()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action += max(self.epsilon, self.epsilon_min) * self.randomer.noise()
        action = np.clip(action, -1.0, 1.0)      # the range of the action should be check again
        # action = np.float32(action.flatten())

        return action

    def load_weights(self, output):
        if output is None:
            return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))
    
    def reset(self):
        self.randomer.reset()

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)
    ##########################
    # Saving the logger and configuration
    # Comment: 
    ##########################
    '''
    def save_config(self, output, save_obj=False):

        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

        if save_obj:
            file = open(output + '/config.obj', 'wb')
            pickle.dump(self.config, file)
            file.close()

    def save_checkpoint(self, ep, total_step, output):

        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)

        torch.save({
            'episodes': ep,
            'total_step': total_step,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, '%s/checkpoint_ep_%d.tar'% (checkpath, ep))


    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        episode = checkpoint['episodes']
        total_step = checkpoint['total_step']
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        return episode, total_step
    '''
