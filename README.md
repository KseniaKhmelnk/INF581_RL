# Car Racing. Projet INF581


## Car Racing with Cross-Entropy Method (CEM) using Gymnasium

This project demonstrates the application of the Cross-Entropy Method (CEM), a reinforcement learning algorithm, to solve the car racing task provided by the `gymnasium` package. The car racing environment is a challenging task that involves controlling a car to navigate through a track as quickly as possible.

## Environment

The car racing environment, known as `CarRacing-v2`, is a part of the `gymnasium` package. It presents a simple car racing simulator, where the objective is to control a car to complete laps on a procedurally generated track. The agent must learn to control the throttle, brake, and steering to navigate the track efficiently.

### Action Space

If continuous there are 3 actions :

0. steering, -1 is full left, +1 is full right

1. gas

2. breaking

If discrete there are 5 actions:

0. do nothing

1. steer left

2. steer right

3. gas

4. brake

### Observation Space

A top-down 96x96 RGB image of the car and race track.

### Rewards

The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

### Starting State

The car starts at rest in the center of the road.

### Episode Termination

The episode finishes when all the tiles are visited. The car can also go outside the playfield - that is, far off the track, in which case it will receive -100 reward and die.

For more details on the `gymnasium` package and the car racing environment, please refer to the official documentation: [Gymnasium Documentation](https://www.gymlibrary.dev/).


## Cross-Entropy Method (CEM)

The Cross-Entropy Method (CEM) is a probabilistic technique used for optimization and solving reinforcement learning problems. It iteratively samples solutions, evaluates them, and selects the best solutions to update the distribution of the next solution samples.

### Mathematical Overview

The goal of CEM is to optimize a policy $\pi_\theta$ parameterized by $\theta$, to maximize expected rewards $R$. The method involves two main steps: sampling and updating.

1. **Sampling**: In each iteration, sample $N$ policies $\theta_i$ from a distribution $p_\theta(*)$, and evaluate their performance in the environment.

2. **Updating**: Select the top $M$ performers $M < N$, and use their parameters to update the distribution $p_\theta(*)$ for the next iteration.

The parameter distribution is often chosen to be a Gaussian $\mathcal{N}(\mu, \Sigma)$, where $\mu$ and $\Sigma$ are updated at each iteration based on the selected top performers.

#### Distribution Update Formulas

Given the top $M$ performing parameter sets, the update formulas for $\mu$ and $\Sigma$ are:

- New mean $\mu_{new} = \frac{1}{M} \sum_{i=1}^{M} \theta_i$

- New covariance $\Sigma_{new} = \frac{1}{M} \sum_{i=1}^{M} (\theta_i - \mu_{new})(\theta_i - \mu_{new})^T$

Where $\theta_i$ are the parameters of the top $M$ policies.

### Application to Car Racing

In the context of the car racing environment, the policy $\pi_\theta$ controls the actions of the car (brake, steering) based on the current state of the environment. The CEM iteratively improves this policy by selecting the parameters that lead to the highest rewards, effectively learning to navigate the car through the track efficiently.
