import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

def preprocess(img):
    # img = cv.resize(img, dsize=(84, 84)) # or you can simply use rescaling
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY) / 255.0

    # gray scale
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # crop
    img = img[:84, 6:90]
    return img 


class CustomEnv(gym.Wrapper):
    def __init__(
        self,
        env,
        skip_frames=4,
        stack_frames=4,
        initial_no_op=50,
        max_episode_steps=10000, 
        **kwargs
    ):
        super(CustomEnv, self).__init__(env, **kwargs)
        
        # image modifications
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(stack_frames, 84,84), dtype=np.uint8)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

        # max episode length
        self.max_episode_steps = max_episode_steps 
        self.step_count = 0
    
    def reset(self, **kwargs):
        # Reset the original environment.
        s, info = self.env.reset(**kwargs)
        self.step_count = 0

        # Do nothing for the next self.initial_no_op steps
        for i in range(self.initial_no_op):
            if self.env.unwrapped.continuous:
                action = [0,0,0] 
            else:
                action = 0
            s, r, terminated, truncated, info = self.env.step(action)
        
        
        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)
        
        
        # The initial observation is simply a copy of the frame s
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info
    
    def step(self, action):        
        self.step_count += 1

        # We take an action for self.skip_frames steps
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            # clip reward
            reward += np.clip(r, a_min=None, a_max=1.0)
            # reward += r
            if terminated or truncated:
                break
        
        if self.step_count >= self.max_episode_steps:
            truncated = True
            info['truncated'] = True


        # Convert a frame to 84 X 84 gray scale one
        s = preprocess(s)

        # Push the current frame s at the end of self.stacked_state
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)

        return self.stacked_state, reward, terminated, truncated, info

    
if __name__ == '__main__':
    version = "v5"
    num_envs = 8
    total_timesteps = 1e7 
    
    make_env = lambda : CustomEnv(gym.make('CarRacing-v2', continuous=False, render_mode="rgb_array"))
    eval_env = DummyVecEnv([make_env])

    envs = SubprocVecEnv([make_env]*num_envs)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./logs/best_model_{version}/', log_path=f'./logs/eval_{version}/', eval_freq=10000)

    model = PPO("CnnPolicy", envs, verbose=0, tensorboard_log="tensorboard")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, tb_log_name="ppo_"+version)

    model.save("ppo-carracing-" + version)
