import numpy as np
import torch
import torch.nn.functional as F


def sample_init_state():
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    theta = np.random.uniform(-np.pi, np.pi)
    return np.array([x, y, theta], dtype=np.float32)

def simulate_system(system, t):
    N = t.shape[0]
    states = np.zeros((3, N), dtype=np.float32)
    controls = np.zeros((2, N-1), dtype=np.float32)
    P = np.zeros((3, 2, N), dtype=np.float32)
    P_hat = np.zeros((3, 2, N), dtype=np.float32)

    states[:, 0] = system.X
    P[:,:,0] = system.P
    P_hat[:,:,0] = system.P_hat
    for i in range(1, N):
        controls[:, i-1] = system.controller(t[i-1])
        states[:, i], P[:,:,i], P_hat[:,:,i] = system.step(controls[:, i-1], t[i-1])

    return states, controls, P, P_hat

def preprocess_control_inputs(inputs):
    max_len = max(input.shape[0] for input in inputs)

    # Pad each array with zeros on the right
    padded_inputs = []
    for input in inputs:
        pad_size = max_len - input.size(0)
        padded_input = F.pad(input, (0, pad_size), mode='constant', value=0)
        padded_inputs.append(padded_input)

    return torch.stack(padded_inputs)