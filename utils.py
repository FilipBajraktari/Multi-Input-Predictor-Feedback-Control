import numpy as np
import torch
import torch.nn.functional as F

from system import System


def sample_init_state():
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    theta = np.random.uniform(-np.pi, np.pi)
    return np.array([x, y, theta])

def simulate_system(sys: System, t):
    N = t.shape[0]
    states = np.zeros((N, 3))
    controls = np.zeros((N-1, 2))

    states[0] = sys.X
    for i in range(1, N):
        controls[i-1] = sys.controller(t[i-1])
        states[i], _ = sys.step(controls[i-1], t[i-1])

    return states, controls

def preprocess_control_inputs(inputs):
    max_len = max(input.shape[0] for input in inputs)

    # Pad each array with zeros on the right
    padded_inputs = []
    for input in inputs:
        pad_size = max_len - input.size(0)
        padded_input = F.pad(input, (0, pad_size), mode='constant', value=0)
        padded_inputs.append(padded_input)

    return torch.stack(padded_inputs)