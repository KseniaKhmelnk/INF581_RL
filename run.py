import argparse
from carRacing.utils import get_model_env_by_name, gif_episode, render_episode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a CarRacing model")
    parser.add_argument("--model", type=str, choices=["DQN", "PPO", "CEM", "DDPG"], help="Choose an algorithm", required=True)
    parser.add_argument("--load-path", type=str, help="Path to the saved model", required=True)
    parser.add_argument("--mode", choices=["play", "gif"], help="Choose how to run", default="gif")
    args = parser.parse_args()
    
    # args
    load_path = args.load_path
    model_name = args.model
    render_mode = "rgb_array" if args.mode == "gif" else "human"

    # get model
    model, env = get_model_env_by_name(model_name, render_mode)

    # load
    model.load(load_path)

    if args.mode == "play":
        render_episode(env, model)
    else: # args.mode == "gif"
        gif_episode(env, model)
