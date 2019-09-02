import torch
import os
from datetime import datetime
import numpy as np
from Replay_Buffer import ReplayBuffer
from Net_Models import Actor, Critic
from Hindsight_Experience_Replay import HER
from mpi_utils import sync_networks, sync_grads
from Normalization import Normalizer
from mpi4py import MPI


###########################################################################
# Name: Deep Deterministic Policy Gradient(DDPG) Agent
# Function: The reinforcement learning agent
# Comment:
###########################################################################
class DDPGAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params

        # _build up the actor/critic evaluated network
        self.actor_net = Actor(env_params, hidden_units=256)
        self.critic_net = Critic(env_params, hidden_units=256)

        # sync the networks across the cpus for parallel training (when running at workstation)
        sync_networks(self.actor_net)
        sync_networks(self.critic_net)

        # _build up the actor/critic target network
        self.actor_target_net = Actor(env_params, hidden_units=256)
        self.critic_target_net = Critic(env_params, hidden_units=256)

        # if gpu is used
        if self.args.cuda:
            self.actor_net.cuda()
            self.critic_net.cuda()
            self.actor_target_net.cuda()
            self.critic_target_net.cuda()

        # the optimizer of the networks
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.args.learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.args.learning_rate_critic)

        # HER sample function
        self.her_sample = HER(self.args.replay_strategy, self.args.replay_ratio, self.env.compute_reward)

        # experience buffer
        self.exp_buffer = ReplayBuffer(self.env_params, self.args.buffer_size, self.her_sample.her_sample_transitions)

        # the normalization of the observation and goal
        self.obs_norm = Normalizer(size=env_params['obs'], clip_range=self.args.clip_range)
        self.goal_norm = Normalizer(size=env_params['d_goal'], clip_range=self.args.clip_range)

        # create the dictionary to save the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            # get the model path
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    ###############################
    # Name: learning
    # Function: Training the model
    # Comment:
    ###############################
    def learning(self):
        success_rate_history = []
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                exp_obs_buff, exp_a_goal_buff, exp_d_goal_buff, exp_actions_buff = [], [], [], []
                for _ in range(self.args.num_exp_per_mpi):
                    # reset the environment and experience
                    exp_obs, exp_a_goal, exp_d_goal, exp_actions = [], [], [], []
                    observations = self.env.reset()
                    obs = observations['observation']
                    a_goal = observations['achieved_goal']
                    d_goal = observations['desired_goal']

                    # interact with the environment
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            input_tensor = self._pre_process_inputs(obs, d_goal)
                            policy_predictions = self.actor_net(input_tensor)
                            action = self._choose_action(policy_predictions)

                        # get the observations from the action
                        observations_next, _, _, info = self.env.step(action)
                        obs_next = observations_next['observation']
                        a_goal_next = observations_next['achieved_goal']
                        exp_obs.append(obs.copy())
                        exp_a_goal.append(a_goal.copy())
                        exp_d_goal.append(d_goal.copy())
                        exp_actions.append(action.copy())
                        # update the state
                        obs = obs_next
                        a_goal = a_goal_next
                    exp_obs.append(obs.copy())
                    exp_a_goal.append(a_goal.copy())
                    exp_obs_buff.append(exp_obs)
                    exp_a_goal_buff.append(exp_a_goal)
                    exp_d_goal_buff.append(exp_d_goal)
                    exp_actions_buff.append(exp_actions)
                exp_obs_buff = np.array(exp_obs_buff)
                exp_a_goal_buff = np.array(exp_a_goal_buff)
                exp_d_goal_buff = np.array(exp_d_goal_buff)
                exp_actions_buff = np.array(exp_actions_buff)
                # store the transitions
                self.exp_buffer.store_transition([exp_obs_buff, exp_a_goal_buff, exp_d_goal_buff, exp_actions_buff])
                self._update_normalizer([exp_obs_buff, exp_a_goal_buff, exp_d_goal_buff, exp_actions_buff])
                for _ in range(self.args.n_batches):
                    self._update_network()      # training the network
                # soft update the network parameter
                self._soft_update_target_network(self.actor_target_net, self.actor_net)
                self._soft_update_target_network(self.critic_target_net, self.critic_net)
            # start evaluation
            success_rate = self._evaluate_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.obs_norm.mean, self.obs_norm.std, self.goal_norm.mean, self.goal_norm.std,
                            self.actor_net.state_dict()], self.model_path + '/model.pt')
            success_rate_history.append(success_rate)
        success_rate_history = np.array(success_rate_history)
        np.savetxt('Plot_Data/Pen_HER.txt', success_rate_history, fmt='%f', delimiter=',')

    ###############################
    # Name: _pre_process_inputs
    # Function: process the inputs for the actor network
    # Comment:
    ###############################
    def _pre_process_inputs(self, obs, goal):
        obs_norm = self.obs_norm.normalize(obs)
        goal_norm = self.goal_norm.normalize(goal)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, goal_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()

        return inputs

    def _choose_action(self, policy_predictions):
        action = policy_predictions.cpu().numpy().squeeze()
        # create the noise
        action += self.args.noise_epsilon * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        random_action = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'],
                                          size=self.env_params['action'])
        # decide random or not
        action += np.random.binomial(1, self.args.random_epsilon, 1)[0] * (random_action - action)

        return action

    def _update_normalizer(self, experience_buff):
        exp_obs, exp_a_goal, exp_d_goal, exp_actions = experience_buff
        exp_obs_next = exp_obs[:, 1:, :]
        exp_a_goal_next = exp_a_goal[:, 1:, :]
        num_exps = exp_actions.shape[1]
        buffer_temp = {'obs': exp_obs,
                       'a_goal': exp_a_goal,
                       'd_goal': exp_d_goal,
                       'actions': exp_actions,
                       'obs_next': exp_obs_next,
                       'a_goal_next': exp_a_goal_next,
                       }
        transitions = self.her_sample.her_sample_transitions(buffer_temp, num_exps)
        obs, d_goal = transitions['obs'], transitions['d_goal']
        transitions['obs'], transitions['d_goal'] = self._pre_process_obs_goal(obs, d_goal)
        # update
        self.obs_norm.update(transitions['obs'])
        self.goal_norm.update(transitions['d_goal'])
        # recompute the stats
        self.obs_norm.recompute_stats()
        self.goal_norm.recompute_stats()

    ###############################
    # Name: _pre_process_obs_goal
    # Function: process the observation and desired goal for the normalization
    # Comment:
    ###############################
    def _pre_process_obs_goal(self, obs, goal):
        obs_proceed = np.clip(obs, -self.args.clip_obs, self.args.clip_obs)
        goal_proceed = np.clip(goal, -self.args.clip_obs, self.args.clip_obs)

        return obs_proceed, goal_proceed

    ###############################
    # Name: _soft_update_target_network
    # Function: soft update the parameters of the target network
    # Comment:
    ###############################
    def _soft_update_target_network(self, target_net, eval_net):
        for target_param, param in zip(target_net.parameters(), eval_net.parameters()):
            target_param.data.copy_((1 - self.args.avg_coeff) * param.data + self.args.avg_coeff * target_param.data)

    ###############################
    # Name: _update_network
    # Function: train the parameters of the actor network and critic network
    # Comment:
    ###############################
    def _update_network(self):
        # sample the transitions
        transitions = self.exp_buffer.sample(self.args.batch_size)
        obs, obs_next, d_goal = transitions['obs'], transitions['obs_next'], transitions['d_goal']
        transitions['obs'], transitions['d_goal'] = self._pre_process_obs_goal(obs, d_goal)
        transitions['obs_next'], transitions['d_goal_next'] = self._pre_process_obs_goal(obs_next, d_goal)
        observation_norm = self.obs_norm.normalize(transitions['obs'])
        d_goal_norm = self.goal_norm.normalize(transitions['d_goal'])
        inputs_norm = np.concatenate([observation_norm, d_goal_norm], axis=1)

        observation_next_norm = self.obs_norm.normalize(transitions['obs_next'])
        d_goal_next_norm = self.goal_norm.normalize(transitions['d_goal_next'])
        inputs_next_norm = np.concatenate([observation_next_norm, d_goal_next_norm], axis=1)

        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        reward_tensor = torch.tensor(transitions['reward'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            reward_tensor = reward_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            actions_next = self.actor_target_net(inputs_next_norm_tensor)
            q_next_value = self.critic_target_net(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = reward_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            clip_return = 1 / (1 - self.args.gamma)      # ??????????????
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # calculate the loss
        real_q_value = self.critic_net(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # the actor loss
        actions_real = self.actor_net(inputs_norm_tensor)
        actor_loss = -self.critic_net(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()

        # start to train the network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_net)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_net)
        self.critic_optimizer.step()

    ###############################
    # Name: _evaluate_agent
    # Function: evaluate the agent
    # Comment:
    ###############################
    def _evaluate_agent(self):
        all_success_rate = []
        for _ in range(self.args.n_eval):
            per_success_rate = []
            observations = self.env.reset()
            obs = observations['observation']
            d_goal = observations['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._pre_process_inputs(obs, d_goal)
                    policy_predictions = self.actor_net(input_tensor)
                    action = policy_predictions.detach().cpu().numpy().squeeze()
                observations_next, _, _, info = self.env.step(action)
                obs = observations_next['observation']
                d_goal = observations_next['desired_goal']
                per_success_rate.append(info['is_success'])
            all_success_rate.append(per_success_rate)
        all_success_rate = np.array(all_success_rate)
        local_success_rate = np.mean(all_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)

        return global_success_rate / MPI.COMM_WORLD.Get_size()
