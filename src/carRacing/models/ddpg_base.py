import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = x.view(-1, 1).T
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64 + action_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, xs):
        height, width = 96, 96
        s, a = xs
        s = s.view(-1, 1).T.to('cuda')
        out = self.fc1(s)
        out = self.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


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


criterion = nn.MSELoss()


class DDPG(object):
    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)

        
        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())  
        self.target_actor.eval()

        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict()) 
        self.target_critic.eval()

        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.target_actor.to('cuda')
        self.target_critic.to('cuda')
        self.critic.to('cuda')
        self.actor.to('cuda')

       
        self.replay_buffer = ReplayBuffer(capacity=10000)


       
        self.batch_size = 5
        self.tau = 0.001
        self.gamma = 0.1

        #
        self.epsilon = 1.0
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True

    def update_policy(self):
        # Sample batch
        batch_info = self.replay_buffer.sample(self.batch_size)

        state_batch = batch_info[0][0]
        action_batch = batch_info[0][1]
        reward_batch = batch_info[0][3]
        next_state_batch = batch_info[0][2]
        terminal_batch = batch_info[0][4]

        #print(reward_batch)
        next_state_batch_np = np.array([np.array(item) for item in next_state_batch])


        # Prepare for the target q batch
        with torch.no_grad():
            action_next = self.target_actor(to_tensor(next_state_batch, device = 'cuda'))
            next_q_values = self.target_critic([to_tensor(next_state_batch, device='cuda'), action_next])
            next_q_values.volatile = False

        target_q_batch = to_tensor(np.array(reward_batch), device = 'cuda') + self.gamma * to_tensor(np.array(terminal_batch, dtype=float), device = 'cuda') * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(state_batch, device = 'cuda'), to_tensor(action_batch, device = 'cuda').unsqueeze(0)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch, device = 'cuda'),
            self.actor(to_tensor(state_batch, device = 'cuda'))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Target update
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

    def random_action(self):
        action = np.random.uniform(-1., 1., self.action_dim)
        self.a_t = action
        return action

    def save_model(self):
        torch.save(
            self.actor.state_dict(),
            'actor.pth')
        torch.save(
            self.critic.state_dict(),
            'critic.pth')



def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, device = 'cuda', volatile=False, requires_grad=False, dtype=torch.float32):
    return Variable(
        torch.tensor(ndarray, device = device, dtype = dtype), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

