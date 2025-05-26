import numpy as np

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