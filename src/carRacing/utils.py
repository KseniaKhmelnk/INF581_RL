import pygame
import imageio

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
                if event.key == pygame.K_ESCAPE:
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