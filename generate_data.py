import os
from datetime import datetime

import h5py
import numpy as np

from unicycle import Unicycle
from utils import sample_init_state, simulate_system

# Simulation Params
dt = 1e-4
T = 10
t = np.arange(0, T + dt, dt)
N = t.shape[0]
delays = [0.5, 0.5]
NDs = [int(delay / dt) for delay in delays]

def main():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    # filename = f"data/const_delay_{timestamp}.h5"
    filename = f"data/const_delay.h5"

    with h5py.File(filename, 'w') as f:
        sample_cnt = 0
        nmb_of_sim = 1
        for _ in range(nmb_of_sim):

            # Execute simulation
            X0 = sample_init_state()
            init_inputs = [np.zeros(NDi) for NDi in NDs]
            sys = Unicycle(X0, init_inputs, dt)
            states, controls = simulate_system(sys, t)

            # Extract data
            nmb_of_ind = 100
            indexes = np.random.randint(NDs[-1], N-NDs[-1], nmb_of_ind)
            for i in indexes:
                group = f.create_group(f'sample_{sample_cnt:04d}')
                sample_cnt += 1

                group.create_dataset('X', data=states[i])
                for k in range(len(NDs)):
                    group.create_dataset(f'U{k}', data=controls[i-NDs[k]:i, k])
                group.create_dataset('P', data=states[i+NDs[-1]])

if __name__ == "__main__":
    main()