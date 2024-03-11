from carRacing.models.ppo import PPO
from carRacing.envs.ppo import envPPO
from carRacing.utils import *

env = envPPO()
model = PPO(env)
model.load('example')

render_episode(env, model)
