import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################################################
# Name: Actor Network
# Function: decide the next action
# Comment: input: [observation, goal],
# output: action(s' probability)
###########################################################################
class Actor(nn.Module):
    def __init__(self, env_params, hidden_units):
        super(Actor, self).__init__()
        self.hidden_units = hidden_units
        self.action_max = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['d_goal'], self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.output_layer = nn.Linear(self.hidden_units, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.action_max * torch.tanh(self.output_layer(x))

        return actions


###########################################################################
# Name: Critic Network
# Function: evaluate the action value
# Comment: input: [observation, goal, actions],
# output: the value of the action
###########################################################################
class Critic(nn.Module):
    def __init__(self, env_params, hidden_units):
        super(Critic, self).__init__()
        self.hidden_units = hidden_units
        self.action_max = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['d_goal'] + env_params['action'], self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.output_layer = nn.Linear(self.hidden_units, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.action_max], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_value = self.output_layer(x)

        return action_value
