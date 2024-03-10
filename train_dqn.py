import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from IPython import display
import imageio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(img):
    img = img[:84, 6:90] # CarRacing-v2-specific cropping
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) / 255.0
    return img


class EnvCarRacing(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        **kwargs):
        super(EnvCarRacing, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
    
    def reset(self):
        # Reset the original environment.
        s, info = self.env.reset()

        # Do nothing for the next `self.initial_no_op` steps
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(0)
        
        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # The initial observation is simply a copy of the frame `s`
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)


            # clip reward
            #if self.clip_reward:
            #    r = np.clip(r, a_min=None, a_max=1.0)
                
            reward += r
            if terminated or truncated:
                break

        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame `s` at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info


class CustomEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        discretize=False,
        stack_frames=4, # number of frames to stack
        skip_frames=4, # number of frames to wait between actions
        initial_no_op=50, # used to skip the 'cinematics' at the start the game
        clip_reward=True, 
        max_episode_steps=700,
        # max_steps_out=250,
        **kwargs
    ):
        super(CustomEnv, self).__init__(env, **kwargs)
 
        # custom observation
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(stack_frames, 84,84), dtype=np.uint8)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.clip_reward = clip_reward
            
        # custom action
        self.discretize = discretize
        if self.discretize:
            assert self.env.unwrapped.continuous
            self.action_map = self._generate_action_map()
            self.action_space = gym.spaces.Discrete(len(self.action_map))
        
        # max episode length
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        # self.max_steps_out = max_steps_out 
        # self.step_out_count = 0
 
 
    def reset(self, **kwargs):
        # reset the original environment.
        s, info = self.env.reset(**kwargs)
        self.step_count = 0
        # self.step_out_count = 0
 
        # do nothing for the next self.initial_no_op steps
        # (wait the 'cinematics')
        for i in range(self.initial_no_op):
            if self.env.unwrapped.continuous:
                action = [0,0,0] 
            else:
                action = 0
            s, r, terminated, truncated, info = self.env.step(action)
        
        
        # gray scale and crop 
        s = self._process_image(s)
        
        # the initial observation is simply a replication of the initial frame
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [self.stack_frames, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):
        if self.discretize == True:
            action = self.action_map[action]
        self.step_count += 1
        # self.step_out_count += 1
 
        # $action is taken for the next $skip_frames frames
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            # reset step out count (car came back)
            # if r > -0.1:
                # self.step_out_count = 0
            # clip reward
            if self.clip_reward:
                r = np.clip(r, a_min=None, a_max=1.0)
            
            reward += r
            if terminated or truncated: 
                break
        
        if self.step_count >= self.max_episode_steps :
        # or self.step_out_count > self.max_steps_out:
            truncated = True
            info['max_episode_steps'] = True
 
 
        # gray scale and crop
        s = self._process_image(s)
 
        # push the current frame s at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
 
        return self.stacked_state, reward, terminated, truncated, info
 
    @staticmethod
    def _process_image(img : np.ndarray) -> np.ndarray:
        # gray scale
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
 
        # crop
        img = img[:84, 6:90]
        return img
            
    @staticmethod
    def _generate_action_map() -> list:
        # default used by CarRacing-v2: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
        afk = np.array([0, 0, 0])
        turn_left = np.array([-1, 0, 0])
        turn_right = np.array([1, 0, 0]) 
        gas = np.array([0, 0.2, 0])
        brake = np.array([0, 0, 0.8])
            
        actions = [
            afk,
            turn_left,
            turn_right,
            gas,
            brake
        ]
        return actions
    
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
    
class DQN:
    def __init__(self,
                state_dim,
                action_dim,
                lr=0.00025,
                epsilon=1.0,
                epsilon_min=0.1,
                gamma=0.99,
                batch_size=32,
                warmup_steps=5000,
                buffer_size=int(1e5),
                target_update_interval=10000):

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
        self.device = torch.device('cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)
        
        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e6
    
    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a
    
    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))
        
        next_q = self.target_network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        result = {'total_steps': self.total_steps, 'value_loss': loss.item()}
        return result
    
    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            
        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result


def evaluate(n_evals=5):
    eval_env = gym.make('CarRacing-v2', continuous=False)
    eval_env = EnvCarRacing(eval_env)
    
    scores = 0
    for i in range(n_evals):
        (s, _), done, ret = eval_env.reset(), False, 0
        while not done:
            a = agent.act(s, training=False)
            s_prime, r, terminated, truncated, info = eval_env.step(a)
            s = s_prime
            ret += r
            done = terminated or truncated
        scores += ret
    return np.round(scores / n_evals, 4)

if __name__ == "__main__":
    env = gym.make('CarRacing-v2', continuous=False)
    env = EnvCarRacing(env)

    max_steps = int(1e6)
    eval_interval = 10000
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)

    history = {'Step': [], 'Reward': []}

    (s, _) = env.reset()
    while True:
        a = agent.act(s)
        s_prime, r, terminated, truncated, info = env.step(a)
        result = agent.process((s, a, r, s_prime, terminated)) 
        
        s = s_prime
        if terminated or truncated:
            s, _ = env.reset()
            
        if agent.total_steps % eval_interval == 0:
            ret = evaluate()
            history['Step'].append(agent.total_steps)
            history['Reward'].append(ret)
            torch.save(agent.network.state_dict(), 'dqn.pt')
        
        if agent.total_steps > max_steps:
            break

    
    model_path = 'dqn.pt'

    # Create the environment
    env = gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array")
    env = EnvCarRacing  (env)

    # Define state and action dimensions
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = DQN(state_dim, action_dim)

    # Load the trained model weights
    agent.network.load_state_dict(torch.load(model_path))

    frames = []
    scores = 0
    (s, _), done, ret = env.reset(), False, 0
    while not done:
        frames.append(env.render())
        a = agent.act(s, training=False)
        s_prime, r, terminated, truncated, info = env.step(a)
        s = s_prime
        ret += r
        done = terminated or truncated
    scores += ret
    print("Reward:", scores)

    img = plt.imshow(frames[0])
    for frame in frames:
        img.set_data(frame) 
        display.display(plt.gcf())
        display.clear_output(wait=True)
    
    
    imageio.mimsave('car-trained-v2.gif', frames)