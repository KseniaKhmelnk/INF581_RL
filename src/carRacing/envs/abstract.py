__all__ = ["CustomEnv"]
from abc import ABC, abstractmethod
import gymnasium as gym


class CustomEnv(ABC, gym.Wrapper):
    def __init__(self, env, **kwargs):
        super(CustomEnv, self).__init__(env, **kwargs)

    @abstractmethod
    def step(self, action):
        ...

    @abstractmethod
    def reset(self):
        ...

    @property
    @abstractmethod
    def observation_space(self):
        ...

    @property
    @abstractmethod
    def action_space(self):
        ...
