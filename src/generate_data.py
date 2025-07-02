from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import tomli
from tqdm import tqdm

from unicycle import SimulationConfig, Unicycle, simulate_system


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

    # Extract simulation config data
    dt = simulation_cfg.dt
    dx = simulation_cfg.dx
    delays = simulation_cfg.delays
    N = simulation_cfg.N
    NX = simulation_cfg.NX

    # Extract dataset generation config data
    num_sim = dataset_cfg.num_sim
    sample_ratio = dataset_cfg.sample_ratio

    # Generate data
    abs_dataset_path_P1 = (Path(__file__).parent.parent / f'data/{dataset_cfg.filename}_P1.h5').resolve()
    abs_dataset_path_P2 = (Path(__file__).parent.parent / f'data/{dataset_cfg.filename}_P2.h5').resolve()
    with (
        h5py.File(abs_dataset_path_P1, 'w') as f1,
        h5py.File(abs_dataset_path_P2, 'w') as f2
    ):
        files = [f1, f2]

        sample_cnt = 0
        for _ in tqdm(range(num_sim)):

            # Execute simulation
            init_state = sample_init_state()
            unicycle = Unicycle(init_state, simulation_cfg)
            states, controls, P, P_hat, control_pdes = simulate_system(unicycle)

            # Extract data
            indexes = np.random.randint(0, N, int(sample_ratio * N))
            for idx in indexes:
                for i, f in enumerate(files):
                    group = f.create_group(f'sample_{sample_cnt:06d}')
                    group.create_dataset('X', data=P[idx, i, 0])                        # States
                    start = i * NX
                    end = (i+1) * NX
                    group.create_dataset('U', data=control_pdes[idx, :, start:end])     # Control PDEs
                    group.create_dataset('P', data=P[idx, i])                           # Predictors
                    varphi = np.array(
                        [(delays[i] - delays[i-1]) if i > 0 else delays[0]],
                        dtype=np.float32,
                    )
                    group.create_dataset('varphi', data=varphi)                         # varphi
                sample_cnt += 1

        # Add global attributes
        for f in files:
            f.attrs['n_states'] = len(sample_init_state())
            f.attrs['m_inputs'] = len(delays)
            f.attrs['num_points'] = NX
            f.attrs['dt'] = dt
            f.attrs['dx'] = dx
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