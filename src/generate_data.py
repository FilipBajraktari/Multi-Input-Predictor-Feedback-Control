from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tomli
from tqdm import tqdm

from system import SimulationConfig, simulate_system
from unicycle import Unicycle


@dataclass
class DatasetGenerationConfig:
    filename: str
    num_sim: int
    sample_ratio: int


def sample_init_state():
    mean = [1, 1, 1]
    width = [0.2, 0.2, 0.2]
    x = np.random.uniform(mean[0]-width[0], mean[0]+width[0])
    y = np.random.uniform(mean[1]-width[1], mean[1]+width[1])
    theta = np.random.uniform(mean[2]-width[2], mean[2]+width[2])
    return np.array([x, y, theta], dtype=np.float32)


def generate_data(simulation_cfg: SimulationConfig, dataset_cfg: DatasetGenerationConfig):

    # Extract config data
    dt = simulation_cfg.dt
    delays = simulation_cfg.delays
    NDs = simulation_cfg.NDs
    N = simulation_cfg.N
    num_sim = dataset_cfg.num_sim
    sample_ratio = dataset_cfg.sample_ratio

    # Generate data
    abs_dataset_path = (Path(__file__).parent.parent / f'data/{dataset_cfg.filename}.h5').resolve()
    with h5py.File(abs_dataset_path, 'w') as f:

        sample_cnt = 0
        for _ in tqdm(range(num_sim)):

            # Execute simulation
            init_state = sample_init_state()
            init_inputs = [np.zeros(NDi, dtype=np.float32) for NDi in NDs]
            unicycle = Unicycle(init_state, init_inputs, simulation_cfg)
            states, controls, P, P_hat = simulate_system(unicycle)

            # Extract data
            indexes = np.random.randint(0, N, int(sample_ratio * N))
            for i in indexes:
                group = f.create_group(f'sample_{sample_cnt:06d}')
                sample_cnt += 1

                group.create_dataset('X', data=states[i])
                for k in range(len(delays)):
                    start_idx = i - NDs[k]
                    if start_idx < 0:
                        pad_length = -start_idx
                        available_data = controls[:i, k]
                        data = np.concatenate([
                            np.zeros(pad_length, dtype=available_data.dtype),
                            available_data,
                        ])
                    else:
                        data = controls[start_idx:i, k]
                    group.create_dataset(f'U{k}', data=data)
                group.create_dataset('P', data=P[i])

        # Add global attributes
        f.attrs['n_states'] = len(sample_init_state())
        f.attrs['m_inputs'] = len(delays)
        f.attrs['num_points'] = NDs[-1]
        f.attrs['dt'] = dt
        f.attrs['delays'] = delays
        f.attrs['creation_date'] = datetime.now().strftime('%Y_%m_%d_%H:%M:%S')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.toml',
                       help='relative path to config file from project directory')
    args = parser.parse_args()

    abs_cofig_path = (Path(__file__).parent.parent / args.config).resolve()
    with open(abs_cofig_path, 'rb') as f:

        # Extract config data from .toml file
        data = tomli.load(f)
        simulation_cfg = SimulationConfig(**data['simulation'])
        dataset_cfg = DatasetGenerationConfig(**data['dataset'])

        generate_data(simulation_cfg, dataset_cfg)