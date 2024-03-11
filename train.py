import argparse
from carRacing.utils import get_model_env_by_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CarRacing model")
    parser.add_argument("--model", type=str, choices=["DQN", "PPO", "CEM", "DDPG"], help="Choose an algorithm", required=True)
    parser.add_argument("--save-path", type=str, help="Path where model will be saved")
    args = parser.parse_args()

    # args
    save_path = args.save_path if hasattr(args, "save_path") else args.model.lower()
    model_name = args.model

    # get model
    model, _ = get_model_env_by_name(model_name, render_mode="rgb_array")
    
    # train
    print(f"Training {args.model} model...")
    model.train()
    
    # save
    model.save(save_path)
    print(f"Model saved at {save_path}")

    
