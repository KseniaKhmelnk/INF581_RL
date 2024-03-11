from carRacing.models.abstract import CarRacingModel
from carRacing.models.cem_base import CNNPolicy, ObjectiveFunction, cem_uncorrelated
import numpy as np
import torch

class CEM(CarRacingModel):
    def __init__(self, env):
        self.make_env = lambda: env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn_policy = CNNPolicy(self.device)
    
    def train(
      self,
      num_episodes: int = 2,
      max_time_steps: int = 1000
    ) -> None:
        hist_dict = {}
        env = self.make_env()
        objective_function = ObjectiveFunction(env=env,
                                       policy=self.nn_policy,
                                       num_episodes=num_episodes,
                                       max_time_steps=max_time_steps)

        init_mean_array = np.random.random(self.nn_policy.num_params)
        init_var_array = np.ones(self.nn_policy.num_params) * 100.
        
        theta, var, hist_dict = cem_uncorrelated(objective_function=objective_function,
                         policy=self.nn_policy,
                         mean_array=init_mean_array,
                         var_array=init_var_array,
                         max_iterations=10,
                         sample_size=20,
                         elite_frac=0.1,
                         print_every=10,
                         success_score=-600,
                         num_evals_for_stop=None,
                         hist_dict=hist_dict)
        objective_function.env.close()

    def predict(self, observation: np.ndarray) -> int:
        return self.nn_policy(observation)

    def load(self, model_path: str) -> None:
        with open(model_path, "rb") as f:
            theta = np.load(f)
        self.nn_policy.change_weights(theta)
    
    def save(self, model_path: str = "cem.pt") -> None:
        torch.save(self.nn_policy.state_dict(), model_path)
