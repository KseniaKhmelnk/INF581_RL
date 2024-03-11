from carRacing.models.abstract import CarRacingModel
from carRacing.models.ddpg_base import Actor, Critic, ReplayBuffer, Transition
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
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)

    def predict(self, observation: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0)
        action = action.cpu().numpy()    
        action = np.clip(action + np.random.normal(0, 0.1), -self.action_bound, self.action_bound)
        return action
    
    def load(self, model_path: str) -> None:
        self.actor.load_state_dict(torch.load(model_path))

    def save(self, model_path: str = "ddpg.pt") -> None:
        torch.save(self.actor.state_dict(), model_path)

    def train(
      self,
      num_episodes = 100,
      max_steps_per_episode = 5, # Максимальное количество шагов в эпизоде
      init_step = 50,
      batch_size = 5,
      gamma = 0.99,
      tau = 0.001,
      actor_lr = 0.001,
      critic_lr = 0.001,
    ):
        
        env = self.make_env()

        # Определение параметров
        state_dim = self.state_dim
        action_dim = self.action_dim
        action_bound = self.action_bound

        # Определение экземпляров сетей Actor и Critic
        actor = Actor(state_dim, action_dim, action_bound)
        critic = Critic(state_dim, action_dim)


        # Создание целевой модели актора и копирование весов из основной модели
        target_actor = copy.deepcopy(actor)
        target_actor.eval()
        
        target_critic = copy.deepcopy(critic)
        target_critic.eval()

        # Определение оптимизаторов
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        # Определение буфера воспроизведения
        replay_buffer = ReplayBuffer(capacity=10000)

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
                    action = actor(state_tensor).squeeze(0)

                print("Original action:", action)
                action = action.cpu().numpy()    
                action = np.clip(action + np.random.normal(0, 0.1), -action_bound, action_bound)
                

                step_info = env.step(action)
                next_state, reward, done = step_info[0], step_info[1], step_info[2]
                frames.append(env.render())

                print(f"Action: {action}, Reward: {episode_reward}, Done: {done}")

                transition = (state, action, next_state, reward, done)
                replay_buffer.push(transition)
                episode_reward += reward

                if len(replay_buffer.memory) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    batch = Transition(*zip(*batch))

                    state_batch = torch.tensor(batch.state, dtype=torch.float32)
                    action_batch = torch.tensor(batch.action, dtype=torch.float32)
                    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32)
                    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
                    done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

                    with torch.no_grad():
                        target_action_next = target_actor(next_state_batch)
                        target_q_next = target_critic(next_state_batch, target_action_next)
                        target_q = reward_batch + gamma * (1.0 - done_batch) * target_q_next

                    critic_loss = F.mse_loss(critic(state_batch, action_batch), target_q)
                    print(f"Critic loss: {critic_loss.item()}, Step: {step}, Episode: {episode}")

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    predicted_action = actor(state_batch)
                    actor_loss = -critic(state_batch, predicted_action).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                state = next_state
                if done:
                    print("Done")
                    print(f"Episode: {episode + 1}, Reward: {episode_reward}")
                    break

        # Закрытие среды
        env.close()

        self.actor = actor

