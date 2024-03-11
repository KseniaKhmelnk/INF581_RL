from carRacing.envs.abstract import CustomEnv 
from typing import Tuple, Dict
import numpy as np
import gymnasium as gym
import cv2 as cv

class envPPO(CustomEnv):
    def __init__(
      self,
      render_mode="rgb_array",
      stack_frames=4, # number of frames to stack
      skip_frames=4, # number of frames to wait between actions
      initial_no_op=50, # used to skip the 'cinematics'    at the start the game
      clip_reward=False,
      expand_action_space=True,
      max_episode_steps=500,
      **kwargs,
    ) -> None:
        self.env = gym.make('CarRacing-v2', continuous=True, max_episode_steps=np.inf, render_mode=render_mode)
        super(envPPO, self).__init__(self.env, **kwargs)
        
        # custom observation
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.clip_reward = clip_reward
        
        # custom action
        self.action_map = self._generate_action_map(expand_action_space)
        
        # max episode length
        self.max_episode_steps = max_episode_steps 
        self.step_count = 0

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(self.stack_frames, 84,84), dtype=np.uint8)

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.action_map))

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        # reset the original environment.
        s, info = self.env.reset()
        self.step_count = 0

        # do nothing for the next self.initial_no_op steps
        # (wait the 'cinematics')
        for _ in range(self.initial_no_op):
            self.env.step([0,0,0]) #idle
        
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

    @staticmethod
    def _generate_action_map(expand_action_space: bool) -> list:
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

        if expand_action_space:
            # soft/hard acceleration
            actions.extend([
                gas * 0.5,
                brake * 0.5,
            ])

            # soft/hard combinations with turns
            for u in [1, 0.5]:
                for v in [1, 0.5]:
                    actions.extend([
                        u*turn_left + v*gas,
                        u*turn_right + v*gas,
                        u*turn_left + v*brake,
                        u*turn_right + v*brake,
                    ])
        return actions
