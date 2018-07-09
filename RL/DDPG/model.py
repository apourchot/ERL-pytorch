import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug


class Actor(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, actions_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, states_dim, actions_dim, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(states_dim, hidden1)
        self.fc2 = nn.Linear(hidden1+actions_dim, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out
