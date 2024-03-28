import gym
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt

from IPython import display
import imageio

import ddpg

torch.set_default_device('cuda')
device = 'cuda'

# Создание экземпляра окружения
env = gym.make('CarRacing-v2', render_mode='rgb_array')

# Определение параметров
state_dim = 96 * 96 * 3
action_dim = 3
action_bound = 1.0
gamma = 0.99
tau = 0.001

# Обучение
num_episodes = 10
max_steps_per_episode = 5000  # Максимальное количество шагов в эпизоде
init_step = 50

agent = ddpg.DDPG(state_dim, action_dim)

agent.actor.load_state_dict(torch.load('actor.pth'))
agent.critic.load_state_dict(torch.load('critic.pth'))

episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    frames = [env.render()]
    for step in range(init_step):
        env.step([0., 0., 0.])

    for step in range(max_steps_per_episode):
        print(f"Step: {step}, Episode: {episode}")

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = agent.random_action()

        #print("Original action:", action)
        #action = action.cpu().numpy()
        action = np.clip(action + np.random.normal(0, 0.1), -action_bound, action_bound)

        step_info = env.step(action)
        next_state, reward, done = step_info[0], step_info[1], step_info[2]
        frames.append(env.render())

        #print(f"Action: {action}, Reward: {episode_reward}, Done: {done}")

        transition = (state, action, next_state, reward, done)
        agent.replay_buffer.push(transition)
        episode_reward += reward

        if step % 100 == 0:
            agent.save_model()

        if len(agent.replay_buffer.memory) > agent.batch_size:
            agent.update_policy()

        state = next_state
        if done:
           # print("Done")
           # print(f"Episode: {episode + 1}, Reward: {episode_reward}")
            break

    episode_rewards.append(episode_reward)


agent.save_model()
env.close()

plt.plot(episode_rewards)
plt.title("Episode Rewards")
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

img = plt.imshow(frames[0])
for frame in frames:
    img.set_data(frame)
    display.display(plt.gcf())
    display.clear_output(wait=True)

imageio.mimsave('test.gif', frames)