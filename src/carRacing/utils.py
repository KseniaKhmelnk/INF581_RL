import pygame

def render_episode(env, model):
    state, _ = env.reset()[0]
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
