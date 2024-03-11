from carRacing.envs.abstract import CustomEnv 
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym
import cv2 as cv

class envDQN(CustomEnv):
    def __init__(
      self,
      render_mode="rgb_array",
      stack_frames=4, # number of frames to stack
      skip_frames=4, # number of frames to wait between actions
      initial_no_op=50, # used to skip the 'cinematics'    at the start the game
      clip_reward=False,
      max_episode_steps=250,
      **kwargs,
    ) -> None:

        self.env = gym.make('CarRacing-v2', continuous=False, max_episode_steps=np.inf, render_mode=render_mode)
        super(envDQN, self).__init__(self.env, **kwargs)
       
        # custom observation
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.clip_reward = clip_reward
        
        # max episode length
        self.max_episode_steps = max_episode_steps 
        self.step_count = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self.stack_frames, 84,84), dtype=np.uint8)

    @property
    def action_space(self): # not changed
        return self.env.action_space

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        # reset the original environment.
        s, info = self.env.reset(**kwargs)
        self.step_count = 0

        # do nothing for the next self.initial_no_op steps
        # (wait the 'cinematics')
        for _ in range(self.initial_no_op):
            self.env.step(0) #idle
        
        # gray scale and crop 
        s = self._process_image(s)
        
        # the initial observation is simply a replication of the initial frame
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [self.stack_frames, 84, 84]
        return self.stacked_state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = self.action_map[action]
        self.step_count += 1

        # $action is taken for the next $skip_frames frames
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            # clip reward
            if self.clip_reward:
                r = np.clip(r, a_min=None, a_max=1.0)
            reward += r
            if terminated or truncated: 
                break
        
        if self.step_count >= self.max_episode_steps:
            truncated = True
            info['max_episode_reached'] = True


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
