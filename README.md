# Car Racing. Projet INF581

**The DQN (Deep Q-Network) algorithm**, developed by DeepMind in 2015, marked a significant breakthrough in artificial intelligence. At its core, the DQN algorithm builds upon the foundation of Q-Learning, a classic reinforcement learning technique. However, it enhances Q-Learning by leveraging deep neural networks and introducing a new concept known as experience replay.
During the training phase of Q-learning, the Q-value of a state-action pair directly is updating this way:

$$Q^{*}_{k + 1} (s, a) \xleftarrow{} Q^{*}_{k}(s, a) + \alpha (r + \gamma 
\max_{a'} \ Q^{*}_{k}(s', a') - Q^{*}_{k}(s, a))$$

For most problems, it is impractical to represent the $Q$-function as a table containing values for each combination of $s$ and $a$. Instead, we train a function approximator, such as a neural network with parameters $\theta$, to estimate the $Q$-values, i.e. $Q(s,a;\theta) \approx Q^{*}(s,a)$
This can done by minimizing the following loss at each step :

$$  L_i (\theta_i) = (y_i - Q(s, a; \theta_i)) ^ 2$$
where $$ y_i = r + \gamma \max_{a'} \ Q_{k}(s', a'; \theta_i).$$
Experience replay memory serves as a crucial component in training the DQN model. It functions by storing transitions observed by the agent during gameplay, allowing for the reuse of this valuable data in subsequent training iterations. This approach significantly enhances the stability and effectiveness of the DQN training procedure, enabling more robust and efficient learning.

The Deep Q-Learning training algorithm has two phases:
- **Sampling**: we perform actions and store the observed experience tuples in a replay memory.
- **Training**: Select a small batch of tuples randomly and learn from this batch using a gradient descent update step.
