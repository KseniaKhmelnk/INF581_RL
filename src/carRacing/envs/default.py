import gymnasium as gym

def Default(continuous: bool, render_mode: str = "rgb_array"):
    return gym.make("CarRacing-v2", continuous=continuous, render_mode=render_mode)
    

