from dataclasses import dataclass
import deepxde as dde
import h5py
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.utils.data import DataLoader, Dataset, random_split


def get_data_loaders(dataset: Dataset, batch_size: int, device: torch.device):

    # Define split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    device_specific_generator = torch.Generator(device=device)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], device_specific_generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=device_specific_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=device_specific_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size, 
        shuffle=False,
        generator=device_specific_generator,
    )

    return train_loader, val_loader, test_loader


class PredictorOperatorDataset(Dataset):

    def __init__(self, file_path, device: torch.device):
        self.file_path = file_path
        self.device = device
        with h5py.File(file_path, 'r') as f:
            self.keys_list = list(f.keys())
            self.n_states = f.attrs['n_states']
            self.m_inputs = f.attrs['m_inputs']
            self.num_points = f.attrs['num_points']
            self.dt = f.attrs['dt']
            self.dx = f.attrs['dx']
            self.delays = f.attrs['delays']
        
    def __len__(self):
        return len(self.keys_list)
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample_key = self.keys_list[idx]
            sample = f[sample_key]
            X = torch.tensor(sample['X'][()], device=self.device)
            U = torch.tensor(sample['U'][()], device=self.device)
            P = torch.tensor(sample['P'][()], device=self.device)
            varphi = torch.tensor(sample['varphi'][()], device=self.device)

        return X, U, P, varphi


@dataclass
class ModelConfig:
    name: str
    path: str


def get_model_class(name):
    if name == 'FNO':
        return FNOProjected
    elif name == 'DeepONet':
        return DeepONetProjected
    elif name == 'FNO+GRU':
        return FNOGRUNet
    elif name == 'DeepONet+GRU':
        return DeepONetGRUNet
    else:
        raise RuntimeError("Model name does not exist!")


class NeuralPredictor(nn.Module):
    
    def __init__(self, n_states, m_inputs, num_points, dt, dx, delays):
        super().__init__()
        self.n_states = n_states
        self.m_inputs = m_inputs
        self.num_points = num_points
        self.dt = dt
        self.dx = dx
        self.delays = delays

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'n_states': self.n_states,
            'm_inputs': self.m_inputs,
            'num_points': self.num_points,
            'dt': self.dt,
            'dx': self.dx,
            'delays': self.delays,
        }, path)

    @classmethod
    def load(cls, path, device='cpu'):
        checkpoint = torch.load(path, weights_only=False)
        model = cls(
            n_states=checkpoint['n_states'],
            m_inputs=checkpoint['m_inputs'],
            num_points=checkpoint['num_points'],
            dt=checkpoint['dt'],
            dx=checkpoint['dx'],
            delays=checkpoint['delays'],
        ).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model


class FNOProjected(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, dx, delays, hidden_size=32):
        super().__init__(n_states, m_inputs, num_points, dt, dx, delays)

        self.fno = FNO(
            n_modes=(12,),  # Number of Fourier modes
            in_channels=n_states+m_inputs+1,
            out_channels=n_states,
            hidden_channels=hidden_size,
        )

    def forward(self, state, control, varphi):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
            - varphi:  (batch_size, 1)

        Returns:
            - prediction: (batch_size, num_points, n)
        """
        combined = torch.cat(
            (
                state.unsqueeze(-1).expand(-1, -1, self.num_points),
                control,
                varphi.unsqueeze(-1).expand(-1, -1, self.num_points),
            ),
            dim=1,
        )
        y = torch.transpose(self.fno(combined), 1, 2)

        return y
    
    def __str__(self):
        return "FNO"


class DeepONetProjected(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, dx, delays, hidden_size=32):
        super().__init__(n_states, m_inputs, num_points, dt, dx, delays)
        self.num_layers = 3
        self.grid = torch.linspace(0, 1, num_points)
        self.grid = self.grid.repeat(n_states+m_inputs+1).reshape(-1, 1)

        # Branch Net
        self.n_input_channel = (n_states + m_inputs + 1) * num_points
        branch_net = [hidden_size] * self.num_layers
        branch_net[0] = self.n_input_channel

        # Trunk Net
        trunk_net = [hidden_size] * self.num_layers
        trunk_net[0] = 1

        self.deeponet = dde.nn.DeepONetCartesianProd(
            branch_net, trunk_net, "relu", "Glorot normal"
        )
        
        # Post-processing to map scalar DeepONet output to vector output_dim
        self.n_output_channel = n_states * num_points
        self.out = torch.nn.Sequential(
            torch.nn.Linear(self.n_input_channel, 4 * self.n_output_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * self.n_output_channel, self.n_output_channel)
        )

    def forward(self, state, control, varphi):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
            - varphi:  (batch_size, 1)

        Returns:
            - prediction: (batch_size, num_points, n)
        """
        combined = torch.cat(
            (
                state.unsqueeze(-1).expand(-1, -1, self.num_points),
                control,
                varphi.unsqueeze(-1).expand(-1, -1, self.num_points),
            ),
            dim=1,
        )                                                           # (batch_size, n+m+1, num_points)
        combined_flat = combined.reshape(combined.shape[0], -1)     # (batch_size, (n+m+1) * num_points)
        
        # DeepONet
        self.grid = self.grid.to(combined_flat.device)
        y = self.deeponet((combined_flat, self.grid))        # (batch_size, (n+m+1) * num_points)
        
        # Post-processing
        y = self.out(y)                                      # (batch_size, num_points * n)
        y = y.reshape(y.shape[0], self.num_points, -1)       # (batch_size, num_points, n)
        
        return y
    
    def __str__(self):
        return "DeepONet"


class FNOGRUNet(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, dx, delays):
        super().__init__(n_states, m_inputs, num_points, dt, dx, delays)

        self.fno = FNOProjected(n_states, m_inputs, num_points, dt, dx, delays)
        self.gru_hidden_size = 32
        self.gru_num_layers = 3
        self.rnn = nn.GRU(n_states, self.gru_hidden_size, self.gru_num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.gru_hidden_size, n_states)

    def forward(self, state, control, varphi):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
            - varphi:  (batch_size, 1)

        Returns:
            - prediction: (batch_size, num_points, n)
        """
        fno_out = self.fno(state, control, varphi)
        y, _ = self.rnn(fno_out)
        y = self.linear(y)

        return y
    
    def __str__(self):
        return "FNO+GRU"


class DeepONetGRUNet(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, dx, delays):
        super().__init__(n_states, m_inputs, num_points, dt, dx, delays)

        self.deeponet = DeepONetProjected(n_states, m_inputs, num_points, dt, dx, delays)
        self.gru_hidden_size = 32
        self.gru_num_layers = 3
        self.rnn = nn.GRU(n_states, self.gru_hidden_size, self.gru_num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.gru_hidden_size, n_states)

    def forward(self, state, control, varphi):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
            - varphi:  (batch_size, 1)

        Returns:
            - prediction: (batch_size, num_points, n)
        """
        deeponet_out= self.deeponet(state, control, varphi)
        y, _ = self.rnn(deeponet_out)
        y = self.linear(y)

        return y
    
    def __str__(self):
        return "DeepONet+GRU"
    

if __name__ == "__main__":
    ...