from carRacing.models.abstract import CarRacingModel
from carRacing.models.ddpg_base import DDPG, Transition
import numpy as np 
import copy 
import torch
import torch.nn.functional as F

class DDPG(CarRacingModel):
    def __init__(self, env, action_bound: float = 1.0):
        self.make_env = lambda: env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_bound = action_bound
        
        multuply = lambda tup : np.array(tup).prod()
        self.state_dim = multuply(env.observation_space.shape)
        self.action_dim = multuply(env.action_space.shape)
        self.action_bound = action_bound
        
        # untrained
        #self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        agent = DDPG(state_dim, action_dim)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.agent.actor(state_tensor).squeeze(0)
        action = action.cpu().numpy()    
        action = np.clip(action + np.random.normal(0, 0.1), -self.action_bound, self.action_bound)
        return action
    
    def load(self, model_path: str) -> None:
        self.agent.actor.load_state_dict(torch.load(model_path))

    def save(self, model_path: str = "ddpg.pt") -> None:
        torch.save(self.agent.actor.state_dict(), model_path)

    def train(
      self,
      num_episodes = 10,
      max_steps_per_episode = 5000, 
      init_step = 50,
      batch_size = 5,
      gamma = 0.99,
      tau = 0.001,
      actor_lr = 0.001,
      critic_lr = 0.001,
    ):
        
        env = self.make_env()

        state_dim = self.state_dim
        action_dim = self.action_dim
        action_bound = self.action_bound

        #actor = Actor(state_dim, action_dim, action_bound)
        #critic = Critic(state_dim, action_dim)
        agent = DDPG(state_dim, action_dim)

        episode_rewards = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0

            frames = [env.render()]
            for step in range(init_step):
                env.step([0.,0.,0.])

            for step in range(max_steps_per_episode):
                print("====================================")
                print(f"Step: {step}, Episode: {episode}")

                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    #action = actor(state_tensor).squeeze(0)
                    action = agent.random_action()

                #print("Original action:", action)
                #action = action.cpu().numpy()    
                #action = np.clip(action + np.random.normal(0, 0.1), -action_bound, action_bound)
                action = np.clip(action + np.random.normal(0, 0.1), -action_bound, action_bound)
                

                step_info = env.step(action)
                next_state, reward, done = step_info[0], step_info[1], step_info[2]
                frames.append(env.render())

                print(f"Action: {action}, Reward: {episode_reward}, Done: {done}")

                transition = (state, action, next_state, reward, done)
                agent.replay_buffer.push(transition)
                episode_reward += reward

                if step % 1000 == 0:
                    agent.save_model()

                if len(agent.replay_buffer.memory) > agent.batch_size:
                    agent.update_policy()

                state = next_state
                if done:
                    #print("Done")
                    print(f"Episode: {episode + 1}, Reward: {episode_reward}")
                    break

            episode_rewards.append(episode_reward)

        agent.save_model()

        env.close()

        plt.plot(episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.show()

        self.agent = agent

