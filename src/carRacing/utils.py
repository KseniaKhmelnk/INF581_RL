import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import imageio

from carRacing.models import DQN, PPO, CEM, DDPG
from carRacing.envs import envDQN, envPPO, Default

def render_episode(env, model):
    assert env.render_mode == "human"
    state, _ = env.reset()
    running = True
    while running:
        action = model.predict(state)
        next_state, _, truncated, terminated, _ = env.step(action)
        state = next_state
        env.render()
        
        # check if episode terminated
        if truncated or terminated:
            running = False

        # check key press
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q: # press ESC or Q to exit
                    running = False
    env.close()

def gif_episode(env, model, save_path: str = "episode.gif", fps=24):
    assert save_path.endswith(".gif")
    assert env.render_mode == "rgb_array"

    state, _ = env.reset()
    imgs = [env.render()]
    running = True
    while running:
        action = model.predict(state)
        next_state, _, truncated, terminated, _ = env.step(action)
        state = next_state
        imgs.append(env.render())
        
        # check if episode terminated
        if truncated or terminated:
            running = False

    env.close()
    imageio.mimsave(save_path, imgs, fps=fps)

def get_model_env_by_name(model_name: str, render_mode: str):
    options = {
        "DQN": (envDQN(render_mode), DQN),
        "PPO": (envPPO(render_mode), PPO),
        "CEM": (Default(False, render_mode), CEM),
        "DDPG": (Default(True, render_mode), DDPG),
    }
    assert model_name in options.keys() 
    env, model_class = options[model_name]
    return model_class(env), env

