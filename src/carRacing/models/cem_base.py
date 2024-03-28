import numpy as np
import gymnasium as gym
import math
import torch
from tqdm import tqdm

class ObjectiveFunction:
    def __init__(self, env, policy, num_episodes=1, max_time_steps=float('inf'), minimization_solver=True):
        self.ndim = policy.num_params
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.max_time_steps = max_time_steps
        self.minimization_solver = minimization_solver

        self.num_evals = 0


    def eval(self, num_episodes=None, max_time_steps=None, render=False, pbar=None, sample_index=None,  reward_threshold=-50):
        """Evaluate a policy"""

        self.num_evals += 1

        if num_episodes is None:
            num_episodes = self.num_episodes

        if max_time_steps is None:
            max_time_steps = self.max_time_steps

        average_total_rewards = 0

        for i_episode in range(self.num_episodes):

            total_rewards = 0.
            state, info = self.env.reset()
            # pbar = tqdm(range(max_time_steps), desc='Evaluating policy')
            for t in range(max_time_steps):
                if render:
                    self.env.render_wrapper.render()
                # print("yes")

                action = self.policy(state)
                state, reward, done, truncated, info = self.env.step(action)
                if done or truncated:
                    break
                total_rewards += reward

                if total_rewards < reward_threshold:
                    break

                if pbar is not None:
                    pbar.set_postfix({
                        'total current reward': total_rewards,
                        'current evaluating time step': t,
                        'current evaluating episode': f"{i_episode}/{num_episodes}",
                        'current sample': sample_index
                    })


            average_total_rewards += float(total_rewards)


            if render:
                print("Test Episode {0}: Total Reward = {1}".format(i_episode, total_rewards))

        average_total_rewards /= self.num_episodes

        if self.minimization_solver:
            average_total_rewards *= -1.

        return average_total_rewards   # Optimizers do minimization by default...


    def __call__(self, num_episodes=None,
                 max_time_steps=None,
                 render=False,
                 pbar=None,
                 sample_index=None):
        return self.eval(num_episodes, max_time_steps, render, pbar, sample_index)


class CNNPolicy(torch.nn.Module):
    def __init__(self, device=None, image_shape=(96, 96, 3), output_dim = 5):
        super(CNNPolicy, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(16 * 12 * 12, 16, bias=True)
        self.fc2 = torch.nn.Linear(16, output_dim, bias=True)
        self.eval=True
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if device is not None:
            self.to(device)
            self.device = device
        else:
            self.device = 'cpu'

    @torch.no_grad()
    def forward(self, current_state):
        current_state = torch.Tensor(current_state).float().view(3, 96, 96).to(self.device)[None, ...]
        current_state = self.pool(torch.nn.functional.relu(self.conv1(current_state)))
        current_state = self.pool(torch.nn.functional.relu(self.conv2(current_state)))
        current_state = self.pool(torch.nn.functional.relu(self.conv3(current_state)))
        current_state = self.flatten(current_state)
        current_state = torch.nn.functional.relu(self.fc1(current_state))
        current_state = self.fc2(current_state)
        return current_state[0].argmax().item()

    def change_weights(self, weights):
        #TODO: overwrite it normally cause if we'd like to change thr architecture of the CNNPolicy it's gonna be impossible -> not flexible at all
        shape_1 = self.conv1.weight.data.shape
        shape_12 = self.conv1.weight.data.shape[0]
        shape_2 = self.conv2.weight.data.shape
        shape_22 = self.conv2.weight.data.shape[0]
        shape_3 = self.conv3.weight.data.shape
        shape_32 = self.conv3.weight.data.shape[0]
        shape_4 = self.fc1.weight.data.shape
        shape_42 = self.fc1.weight.data.shape[0]
        shape_5 = self.fc2.weight.data.shape
        shape_52 = self.fc2.weight.data.shape[0]

        curr_index = 0
        self.conv1.weight.data = torch.Tensor(weights[:math.prod(shape_1)]).view(*tuple(shape_1)).to(self.device)
        curr_index += math.prod(shape_1)
        self.conv1.bias.data = torch.Tensor(weights[curr_index: curr_index + shape_12]).to(self.device)
        curr_index+=shape_12

        self.conv2.weight.data = torch.Tensor(weights[curr_index: curr_index + math.prod(shape_2)]).view(*tuple(shape_2)).to(self.device)
        curr_index += math.prod(shape_2)
        self.conv2.bias.data = torch.Tensor(weights[curr_index : curr_index + shape_22]).to(self.device)
        curr_index += shape_22

        self.conv3.weight.data = torch.Tensor(weights[curr_index: curr_index + math.prod(shape_3)]).view(*tuple(shape_3)).to(self.device)
        curr_index += math.prod(shape_3)
        self.conv3.bias.data = torch.Tensor(weights[curr_index : curr_index + shape_32]).to(self.device)
        curr_index += shape_32

        self.fc1.weight.data = torch.Tensor(weights[curr_index: curr_index + math.prod(shape_4)]).view(*tuple(shape_4)).to(self.device)
        curr_index += math.prod(shape_4)
        self.fc1.bias.data = torch.Tensor(weights[curr_index : curr_index + shape_42]).to(self.device)
        curr_index += shape_42

        self.fc2.weight.data = torch.Tensor(weights[curr_index: curr_index + math.prod(shape_5)]).view(*tuple(shape_5)).to(self.device)
        curr_index += math.prod(shape_4)
        # self.fc2.bias.data = torch.Tensor(weights[curr_index : ]).to(self.device)
        # curr_index += shape_52

def cem_uncorrelated(objective_function,
                     policy,
                     mean_array,
                     var_array,
                     max_iterations=500,
                     sample_size=50,
                     elite_frac=0.2,
                     print_every=10,
                     success_score=float("inf"),
                     num_evals_for_stop=None,
                     hist_dict=None):
    """Cross-entropy method.

    Params
    ======
        objective_function (function): the function to maximize
        mean_array (array of floats): the initial proposal distribution (mean vector)
        var_array (array of floats): the initial proposal distribution (variance vector)
        max_iterations (int): number of training iterations
        sample_size (int): size of population at each iteration
        elite_frac (float): rate of top performers to use in update with elite_frac ∈ ]0;1]
        print_every (int): how often to print average score
        hist_dict (dict): logs
    """
    assert 0. < elite_frac <= 1.

    n_elite = math.ceil(sample_size * elite_frac)
    if len(hist_dict.keys()) == 0:
        hist_dict['reward'] = []

    n_params = len(mean_array)
    pbar = tqdm(range(max_iterations), desc='Fitting theta')
    for i in pbar:
        samples = np.random.normal(mean_array, np.sqrt(var_array), (sample_size, n_params))
        scores = []
        for sample_index in range(len(samples)):
            policy.change_weights(samples[sample_index])
            scores += [objective_function(pbar=pbar, sample_index=sample_index)]
        scores = np.array(scores)
        # print(scores[scores.argsort()])
        elite_idx = scores.argsort()[:n_elite]
        elite_samples = samples[elite_idx]

        mean_array = elite_samples.mean(0)
        # print(mean_array)
        var_array = elite_samples.var(0)


        pbar.set_postfix({
            'current best score': -scores[elite_idx[0]]
        })

        hist_dict['reward']+=[np.mean(-scores)]

        if (i % print_every == 0 or i == max_iterations - 1) and i != 0:
            print(f"""
            =======================
            Iteration {i+1},
            mean score = {-scores.mean()},
            current best score = {-scores[elite_idx[0]]}
            =======================""")

        if scores[elite_idx[0]] <= success_score:
            print(f"\nWe got succes score at {i+1} iteration, current best score = {-scores[elite_idx[0]]}, success_score = {-success_score}")
            break

        if num_evals_for_stop is not None and objective_function.num_evals >= num_evals_for_stop:
            print(f"\nStopping after {objective_function.num_evals} evaluations.")
            break

    return mean_array, var_array, hist_dict


def cem_uncorrelated(objective_function,
                     policy,
                     mean_array,
                     var_array,
                     max_iterations=500,
                     sample_size=50,
                     elite_frac=0.2,
                     print_every=10,
                     success_score=float("inf"),
                     num_evals_for_stop=None,
                     hist_dict=None):
    """Cross-entropy method.

    Params
    ======
        objective_function (function): the function to maximize
        mean_array (array of floats): the initial proposal distribution (mean vector)
        var_array (array of floats): the initial proposal distribution (variance vector)
        max_iterations (int): number of training iterations
        sample_size (int): size of population at each iteration
        elite_frac (float): rate of top performers to use in update with elite_frac ∈ ]0;1]
        print_every (int): how often to print average score
        hist_dict (dict): logs
    """
    assert 0. < elite_frac <= 1.

    n_elite = math.ceil(sample_size * elite_frac)
    if len(hist_dict.keys()) == 0:
        hist_dict['reward'] = []

    n_params = len(mean_array)
    pbar = tqdm(range(max_iterations), desc='Fitting theta')
    for i in pbar:
        samples = np.random.normal(mean_array, np.sqrt(var_array), (sample_size, n_params))
        scores = []
        for sample_index in range(len(samples)):
            policy.change_weights(samples[sample_index])
            scores += [objective_function(pbar=pbar, sample_index=sample_index)]
        scores = np.array(scores)
        # print(scores[scores.argsort()])
        elite_idx = scores.argsort()[:n_elite]
        elite_samples = samples[elite_idx]

        mean_array = elite_samples.mean(0)
        # print(mean_array)
        var_array = elite_samples.var(0)


        pbar.set_postfix({
            'current best score': -scores[elite_idx[0]]
        })

        hist_dict['reward']+=[np.mean(-scores)]

        if (i % print_every == 0 or i == max_iterations - 1) and i != 0:
            print(f"""
            =======================
            Iteration {i+1},
            mean score = {-scores.mean()},
            current best score = {-scores[elite_idx[0]]}
            =======================""")

        if scores[elite_idx[0]] <= success_score:
            print(f"\nWe got succes score at {i+1} iteration, current best score = {-scores[elite_idx[0]]}, success_score = {-success_score}")
            break

        if num_evals_for_stop is not None and objective_function.num_evals >= num_evals_for_stop:
            print(f"\nStopping after {objective_function.num_evals} evaluations.")
            break

    return mean_array, var_array, hist_dict


def cem_correlated(objective_function,
                   policy, 
                   mean_array,
                   var_array,
                   max_iterations=500,
                   sample_size=50,
                   elite_frac=0.2,
                   print_every=10,
                   success_score=float("inf"),
                   num_evals_for_stop=None,
                   hist_dict=None):
    """Cross-entropy method.

    Params
    ======
        objective_function (function): the function to maximize
        mean_array (array of floats): the initial proposal distribution (mean vector)
        var_array (array of floats): the initial proposal distribution (variance vector)
        max_iterations (int): number of training iterations
        sample_size (int): size of population at each iteration
        elite_frac (float): rate of top performers to use in update with elite_frac ∈ ]0;1]
        print_every (int): how often to print average score
        hist_dict (dict): logs
    """
    assert 0. < elite_frac <= 1.

    n_elite = math.ceil(sample_size * elite_frac)
    cov_array = np.diag(var_array)

    if len(hist_dict.keys()) == 0:
        hist_dict['reward'] = []

    n_params = len(mean_array)
    pbar = tqdm(range(max_iterations), desc='Fitting theta')
    for i in pbar:
        samples = np.random.multivariate_normal(mean_array, cov_array, size=sample_size)
        scores = []
        for sample_index in range(len(samples)):
            policy.change_weights(samples[sample_index])
            scores += [objective_function(pbar=pbar, sample_index=sample_index)]
        scores = np.array(scores)
        # print(scores[scores.argsort()])
        elite_idx = scores.argsort()[:n_elite]
        elite_samples = samples[elite_idx]

        mean_array = elite_samples.mean(0)
        # print(mean_array)
        cov_array = np.cov(elite_samples, rowvar=False)


        pbar.set_postfix({
            'current best score': -scores[elite_idx[0]]
        })

        hist_dict['reward']+=[np.mean(-scores)]

        if (i % print_every == 0 or i == max_iterations - 1) and i != 0:
            print(f"""
            =======================
            Iteration {i+1},
            mean score = {-scores.mean()},
            current best score = {-scores[elite_idx[0]]}
            =======================""")

        if scores[elite_idx[0]] <= success_score:
            print(f"\nWe got succes score at {i+1} iteration, current best score = {-scores[elite_idx[0]]}, success_score = {-success_score}")
            break

        if num_evals_for_stop is not None and objective_function.num_evals >= num_evals_for_stop:
            print(f"\nStopping after {objective_function.num_evals} evaluations.")
            break

    return mean_array, var_array, hist_dict


def cem(objective_function,
                   policy, 
                   mean_array,
                   var_array,
                   correlated: bool=False,
                   max_iterations=500,
                   sample_size=50,
                   elite_frac=0.2,
                   print_every=10,
                   success_score=float("inf"),
                   num_evals_for_stop=None,
                   hist_dict=None):
    
    """_summary_

    Returns:
        depending on correlated it returns correlated or uncorrelated cem method
    """   
    
    if correlated:
        return cem_correlated(objective_function,
                   policy, 
                   mean_array,
                   var_array,
                   max_iterations,
                   sample_size,
                   elite_frac,
                   print_every,
                   success_score,
                   num_evals_for_stop,
                   hist_dict)
    else:
        return cem_uncorrelated(objective_function,
                   policy, 
                   mean_array,
                   var_array,
                   max_iterations,
                   sample_size,
                   elite_frac,
                   print_every,
                   success_score,
                   num_evals_for_stop,
                   hist_dict)