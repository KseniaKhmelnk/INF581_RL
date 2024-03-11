from carRacing.models.abstract import CarRacingModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class NeuralNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, activation=F.relu):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2) 
        self.in_features = 32 * 9 * 9
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (torch.FloatTensor(self.s[ind]),
                torch.FloatTensor(self.a[ind]),
                torch.FloatTensor(self.r[ind]),
                torch.FloatTensor(self.s_prime[ind]),
                torch.FloatTensor(self.terminated[ind]))



class DQN(CarRacingModel):

    def __init__(
      self,
      # state_dim,
      # action_dim,
      env,
      lr=0.00025,
      epsilon=1.0,
      epsilon_min=0.1,
      gamma=0.99,
      batch_size=32,
      warmup_steps=5000,
      buffer_size=int(1e5),
      target_update_interval=10000
    ):
        self.make_env = lambda: env

        state_dim = env.observation_space.shape
        action_dim = env.action_space.n

        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = NeuralNetwork(state_dim[0], action_dim)
        self.target_network = NeuralNetwork(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1, ), buffer_size)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    def predict(self, observation: np.ndarray) -> int:
        return self._act(observation, training=False)

    def train(
      self,
      eval_interval: int = 10000,
      max_steps: int = int(1e6),
    ) -> None:
        env = self.make_env()
        history = {'Step': [], 'Reward': []} 
        (s, _) = env.reset()
        while True:
            a = self._act(s)
            s_prime, r, terminated, truncated, info = env.step(a)
            result = self._process((s, a, r, s_prime, terminated)) 
            
            s = s_prime
            if terminated or truncated:
                s, _ = env.reset()
                
            if self.total_steps % eval_interval == 0:
                ret = self._evaluate()
                history['Step'].append(self.total_steps)
                history['Reward'].append(ret)
                torch.save(self.network.state_dict(), './logs/dqn/dqn.pt')
            
            if self.total_steps > max_steps:
                break

    def load(self, model_path: str) -> None:
        self.network.load_state_dict(torch.load(model_path))

    def save(self, model_path: str) -> None:
        torch.save(self.network.state_dict(), model_path)
    
    @torch.no_grad()
    def _act(self, x, training=True) -> int:
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return int(a)

    def _learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        result = {'total_steps': self.total_steps, 'value_loss': loss.item()}
        return result

    def _process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self._learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result

    def _evaluate(self, n_evals=5):
        eval_env = self.make_env()

        scores = 0
        for i in range(n_evals):
            (s, _), done, ret = eval_env.reset(), False, 0
            while not done:
                a = self._act(s, training=False)
                s_prime, r, terminated, truncated, info = eval_env.step(a)
                s = s_prime
                ret += r
                done = terminated or truncated
            scores += ret
        return np.round(scores / n_evals, 4)
