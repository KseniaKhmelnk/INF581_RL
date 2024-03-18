from carRacing.models.abstract import CarRacingModel

import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

import numpy as np

class  PPO(CarRacingModel):
    
    def __init__(self, env):
        self.make_env = lambda: env

    def train(
        self,
        total_timesteps: int = int(1e6),
        num_envs: int = 1,
        eval_freq: int = 0,
        checkpoint_freq: int = 0,
    ) -> None:

        check_env(self.make_env())

        assert isinstance(num_envs, int) and num_envs > 0
        if num_envs > 1: 
            envs = SubprocVecEnv([self.make_env]*num_envs)
        else: 
            envs = DummyVecEnv([self.make_env])
        
        
        callback_list = []
        
        # add eval callback
        assert isinstance(eval_freq, int)
        if eval_freq > 0:
            callback_list.append(
              EvalCallback(
                eval_env=DummyVecEnv([self.make_env]),
                best_model_save_path='./logs/ppo/best_model',
                log_path='./logs/ppo/results',
                eval_freq=eval_freq//num_envs, 
              )
            )
        
        # add checkpoint callback
        assert isinstance(checkpoint_freq, int) 
        if checkpoint_freq > 0:
            callback_list.append(
              CheckpointCallback(
               save_path= './logs/ppo/checkpoints',
               name_prefix="ppo",
               save_freq=checkpoint_freq//num_envs,
              )
            )
        
        # train
        callback = CallbackList(callback_list) 
        self.model = sb3.PPO("CnnPolicy", envs, verbose=0, tensorboard_log="tensorboard")
        self.model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name="ppo")
        
    
    def predict(self, observation: np.ndarray) -> int:
        return self.model.predict(observation)[0]

    def load(self, load_path: str) -> None: 
        self.model = sb3.PPO.load(
                load_path,
                custom_objects = {
                    "learning_rate": 0.0,
                    "lr_schedule": lambda _: 0.0,
                    "clip_range": lambda _: 0.0,
                }) # fix small problem, when loading a model trained with python 3.7 in python3.8+ 
    
    def save(self, save_path: str = "ppo") -> None:
        self.model.save(save_path)
