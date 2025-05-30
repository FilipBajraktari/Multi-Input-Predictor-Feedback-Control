from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm

from unicycle import Unicycle
from utils import sample_init_state, simulate_system

from . import *


def main():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    filename = f"data/const_distinct_delays.h5"

    with h5py.File(filename, 'w') as f:
        sample_cnt = 0
        nmb_of_sim = 1000
        for _ in tqdm(range(nmb_of_sim)):

            # Execute simulation
            X0 = sample_init_state()
            init_inputs = [np.zeros(NDi, dtype=np.float32) for NDi in NDs]
            sys = Unicycle(X0, init_inputs, dt)
            states, controls, P, P_hat = simulate_system(sys, t)

            # Extract data
            nmb_of_small_ind = int(delays[-1] * 100)
            nmb_of_large_ind = int(T * 10)
            small_indexes = np.random.randint(0, NDs[-1], nmb_of_small_ind)
            large_indexes = np.random.randint(NDs[-1], N-NDs[-1], nmb_of_large_ind)
            for i in np.concatenate([small_indexes, large_indexes]):
                group = f.create_group(f'sample_{sample_cnt:04d}')
                sample_cnt += 1

                group.create_dataset('X', data=states[:, i])
                for k in range(len(delays)):
                    start_idx = i - NDs[k]
                    if start_idx < 0:
                        pad_length = -start_idx
                        available_data = controls[k, 0:i]
                        data = np.concatenate([
                            np.zeros(pad_length, dtype=available_data.dtype), 
                            available_data,
                        ])
                    else:
                        data = controls[k, start_idx:i]
                    group.create_dataset(f'U{k}', data=data)
                group.create_dataset('P', data=states[:, i+1:i+NDs[-1]+1])

        # Add global attributes
        f.attrs['n_states'] = len(sample_init_state())
        f.attrs['m_inputs'] = len(delays)
        f.attrs['num_points'] = NDs[-1]
        f.attrs['dt'] = dt
        f.attrs['delays'] = delays
        f.attrs['creation_date'] = timestamp

if __name__ == "__main__":
    main()