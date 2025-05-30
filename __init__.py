import numpy as np

# Simulation Params
dt = 1e-3
T = 10
t = np.arange(0, T + dt, dt)
N = t.shape[0]
delays = [0.25, 0.75]
NDs = [int(delay / dt) for delay in delays]