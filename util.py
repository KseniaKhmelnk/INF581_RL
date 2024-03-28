import numpy as np
import torch
from collections import namedtuple
from torch.autograd import Variable

# Определение класса Transition для хранения переходов
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def to_numpy(var):
    return var.data.numpy()

def to_tensor(ndarray, device = 'cuda', volatile=False, requires_grad=False, dtype=torch.float32):
    return Variable(
        torch.tensor(ndarray, device = device, dtype = dtype), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)