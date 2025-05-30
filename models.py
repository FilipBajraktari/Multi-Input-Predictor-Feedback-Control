import h5py
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.utils.data import DataLoader, Dataset, random_split

from utils import preprocess_control_inputs


class NeuralPredictor(nn.Module):
    def __init__(self, n_states, m_inputs, num_points):
        super().__init__()
        self.n_states = n_states
        self.m_inputs = m_inputs
        self.num_points = num_points

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'n_states': self.n_states,
            'm_inputs': self.m_inputs,
            'num_points': self.num_points,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, weights_only=False)
        model = cls(
            n_states=checkpoint['n_states'],
            m_inputs=checkpoint['m_inputs'],
            num_points=checkpoint['num_points'],
        ).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model

class FNOProjection(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, width=128, modes=20):
        super().__init__(n_states, m_inputs, num_points)

        self.fno = FNO(
            n_modes=(modes,),
            in_channels=self.n_states+self.m_inputs,
            out_channels=self.n_states,
            hidden_channels=width,
        )

    def forward(self, state, controls):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)

        Returns:
            - prediction: (batch_size, n, num_points)
        """
        combined = torch.cat(
            (state.unsqueeze(-1).expand(-1, -1, self.num_points), controls),
            dim=1,
        )
        return self.fno(combined)
    
    def __str__(self):
        return "FNO"


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
            X = torch.tensor(sample['X'][:])
            U = preprocess_control_inputs([
                torch.tensor(sample[f'U{i}'][:]) for i in range(self.m_inputs)
            ])
            P = torch.tensor(sample['P'][:])

        return X, U, P


def get_data_loaders(dataset: Dataset, batch_size: int):

    # Define split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    ...