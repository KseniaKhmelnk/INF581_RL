import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

# Определение класса Transition для хранения переходов
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Преобразование входных данных в плоский формат
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x

# Определение класса Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        # Преобразование тензора действия к нужной размерности
        #print(state.shape, action.shape)
        height, width = 96, 96
        #action = action.unsqueeze(1).unsqueeze(2).expand(state.size(0), height, width, -1)
        state = state.view(state.size(0), -1)

        #print(state.shape, action.shape)
        x = torch.cat([state, action], dim=1)
        #print(x.shape)

        #batch_size = x.size(0)
        #num_features = x.size(1) * x.size(2) * x.size(3)
        #x = x.view(num_features, batch_size)

        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Определение буфера воспроизведения для хранения опыта
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[idx] for idx in indices]

