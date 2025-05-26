import h5py
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import preprocess_control_inputs


class NeuralPredictor(torch.nn.Module):
    def __init__(self, n_state: int, n_input: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_state = n_state
        self.n_input = n_input
        self.seq_len = seq_len

        self.mse_loss = torch.nn.MSELoss()
    
class FNOProjection(torch.nn.Module):
    def __init__(self, n_states, m_inputs, num_points, width=64, modes=16):
        super().__init__()
        self.n_states = n_states
        self.m_inputs = m_inputs
        self.num_points = num_points

        self.fno = FNO(
            n_modes=(modes,),
            in_channels=self.n_states+self.m_inputs,
            out_channels=self.n_states,
            hidden_channels=width,
        )

    def forward(self, state, controls, prediction):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
        """
        combined = torch.cat(
            (state.expand(-1, -1, self.num_points), controls),
            dim=1,
        )
        return self.fno(combined)
    
class PredictorOperatorDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.keys_list = list(f.keys())
            self.n_states = f.attrs['n_states']
            self.m_inputs = f.attrs['m_inputs']
            self.num_points = f.attrs['num_points']
        
    def __len__(self):
        return len(self.keys_list)
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample_key = self.keys_list[idx]
            sample = f[sample_key]
            X = torch.tensor(sample['X'][:], dtype=torch.float32).unsqueeze(-1)
            # U = torch.stack([
            #     torch.tensor(sample[f'U{i}'][:], dtype=torch.float32)
            #     for i in range(self.m_inputs)
            # ])
            U = preprocess_control_inputs([
                torch.tensor(sample[f'U{i}'][:], dtype=torch.float32)
                for i in range(self.m_inputs)
            ])
            P = torch.tensor(sample['P'][:], dtype=torch.float32).unsqueeze(-1)

        return X, U, P

if __name__ == "__main__":
    dataset = PredictorOperatorDataset("data/const_delay.h5")
    model = FNOProjection(
        n_states=dataset.n_states,
        m_inputs=dataset.m_inputs,
        num_points=dataset.num_points,
    )
    X, U, P = dataset[0]
    print(model(X.unsqueeze(0), U.unsqueeze(0), P.unsqueeze(0)))
