import torch
import numpy as np
import torch.nn.functional as fun


###############################################################
# Name: Weight initializer
# Function:
# Comment: known from https://arxiv.org/abs/1502.01852
###############################################################
def param_initialization(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    
    return torch.Tensor(size).uniform_(-v, v)


###############################################################
# Name: Critic Module
# Function:
# Comment: 
###############################################################
class Critic(torch.nn.Module):
    def __init__(self, state_units, action_units, hidden_units_1=128, hidden_units_2=64, eps=0.03):
        super(Critic, self).__init__()
        self.state_units = state_units
        self.action_units = action_units

        self.fc_1 = torch.nn.Linear(state_units, hidden_units_1)
        self.fc_1.weight.data = param_initialization(self.fc_1.weight.data.size())

        self.fc_2 = torch.nn.Linear(hidden_units_1 + action_units, hidden_units_2)
        self.fc_2.weight.data = param_initialization(self.fc_2.weight.data.size())

        self.fc_3 = torch.nn.Linear(hidden_units_2, 1)
        self.fc_3.weight.data.uniform_(-eps, eps)

        self.relu = torch.nn.ReLU()

    ##########################
    # Forward Propagation
    # Comment: return value Q(s,a)
    ##########################
    def forward(self, state, action):
        temp = self.relu(self.fc_1(state))
        x = torch.cat((temp, action), dim=1)
        x = self.relu(self.fc_2(x))
        value = self.fc_3(x)

        return value


###############################################################
# Name: Actor Module
# Function:
# Comment:
###############################################################
class Actor(torch.nn.Module):
    def __init__(self, state_units, action_units, hidden_units_1=128, hidden_units_2=64, eps=0.03):
        super(Actor, self).__init__()
        self.state_units = state_units
        self.action_units = action_units

        self.fc_1 = torch.nn.Linear(state_units, hidden_units_1)
        self.fc_1.weight.data = param_initialization(self.fc_1.weight.data.size())

        self.fc_2 = torch.nn.Linear(hidden_units_1, hidden_units_2)
        self.fc_2.weight.data = param_initialization(self.fc_2.weight.data.size())

        self.fc_3 = torch.nn.Linear(hidden_units_2, action_units)
        self.fc_3.weight.data.uniform_(-eps, eps)

        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    ##########################
    # Forward Propagation
    # Comment: return action [n, action_dim]
    ##########################
    def forward(self, state):
        x = self.relu(self.fc_1(state))
        x = self.relu(self.fc_2(x))
        action = self.tanh(self.fc_3(x))      # tanh limit (-1, 1)
        
        return action
