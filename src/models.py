from dataclasses import dataclass
import deepxde as dde
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO
from torch.utils.data import DataLoader, Dataset, random_split


# Deepxde sets defualt device to 'cuda' if it is available and that breaks
# Dataloader becuase the generator it uses must be on the same device as
# data. GPU random generator is way slower than CPU's. Thus, we override
# torch default device back to CPU
torch.set_default_device('cpu')


def preprocess_control_inputs(inputs):
    max_len = max(input.shape[0] for input in inputs)

    # Pad each array with zeros on the right
    padded_inputs = []
    for input in inputs:
        pad_size = max_len - input.size(0)
        padded_input = F.pad(input, (0, pad_size), mode='constant', value=0)
        padded_inputs.append(padded_input)

    return torch.stack(padded_inputs)


def get_data_loaders(dataset: Dataset, batch_size: int, device: torch.device):

    # Define split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    device_specific_generator = torch.Generator(device=device)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
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
            self.delays = f.attrs['delays']
        
    def __len__(self):
        return len(self.keys_list)
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            sample_key = self.keys_list[idx]
            sample = f[sample_key]
            X = torch.tensor(sample['X'][:], device=self.device)
            U = preprocess_control_inputs([
                torch.tensor(sample[f'U{i}'][:], device=self.device)
                for i in range(self.m_inputs)
            ])
            P = torch.tensor(sample['P'][:], device=self.device)

        return X, U, P


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
    
    def __init__(self, n_states, m_inputs, num_points, dt, delays):
        super().__init__()
        self.n_states = n_states
        self.m_inputs = m_inputs
        self.num_points = num_points
        self.dt = dt
        self.delays = delays

    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'n_states': self.n_states,
            'm_inputs': self.m_inputs,
            'num_points': self.num_points,
            'dt': self.dt,
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
            delays=checkpoint['delays'],
        ).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model


class FNOProjected(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, delays, hidden_size=32):
        super().__init__(n_states, m_inputs, num_points, dt, delays)

        self.fno = FNO(
            n_modes=(12,),  # Number of Fourier modes
            in_channels=n_states+m_inputs,
            out_channels=n_states,
            hidden_channels=hidden_size,
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


class DeepONetProjected(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, delays, hidden_size=32):
        super().__init__(n_states, m_inputs, num_points, dt, delays)
        self.num_layers = 3
        self.grid = torch.arange(0, delays[-1], dt)
        self.grid = self.grid.repeat(n_states+m_inputs).reshape(-1, 1)

        # Branch Net
        self.n_input_channel = (n_states + m_inputs) * num_points
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
        ).transpose(1, 2)   # (batch_size, num_points, n+m)
        combined_flat = combined.reshape(combined.shape[0], -1) # (batch_size, (n+m) * num_points)
        
        # DeepONet
        self.grid = self.grid.to(combined_flat.device)
        y = self.deeponet((combined_flat, self.grid))   # (batch_size, (n+m) * num_points)
        
        # Post-processing
        y = self.out(y)     # (batch_size, (n+m) * num_points)
        y = y.reshape(y.shape[0], self.num_points, -1).transpose(1, 2)  # (batch_size, n, num_points)
        
        return y
    
    def __str__(self):
        return "DeepONet"


class FNOGRUNet(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, delays):
        super().__init__(n_states, m_inputs, num_points, dt, delays)

        self.fno = FNOProjected(n_states, m_inputs, num_points, dt, delays)
        self.gru_hidden_size = 32
        self.gru_num_layers = 3
        self.rnn = nn.GRU(n_states, self.gru_hidden_size, self.gru_num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.gru_hidden_size, n_states)

    def forward(self, state, controls):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)

        Returns:
            - prediction: (batch_size, n, num_points)
        """
        fno_out = self.fno(state, controls)
        y, _ = self.rnn(fno_out.transpose(1, 2))
        y = self.linear(y)

        return y.transpose(1, 2)
    
    def __str__(self):
        return "FNO+GRU"


class DeepONetGRUNet(NeuralPredictor):

    def __init__(self, n_states, m_inputs, num_points, dt, delays):
        super().__init__(n_states, m_inputs, num_points, dt, delays)

        self.deeponet = DeepONetProjected(n_states, m_inputs, num_points, dt, delays)
        self.gru_hidden_size = 32
        self.gru_num_layers = 3
        self.rnn = nn.GRU(n_states, self.gru_hidden_size, self.gru_num_layers, batch_first=True)
        self.linear = torch.nn.Linear(self.gru_hidden_size, n_states)

    def forward(self, state, controls):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)

        Returns:
            - prediction: (batch_size, n, num_points)
        """
        deeponet_out= self.deeponet(state, controls)
        y, _ = self.rnn(deeponet_out.transpose(1, 2))
        y = self.linear(y)

        return y.transpose(1, 2)
    
    def __str__(self):
        return "DeepONet+GRU"
    

if __name__ == "__main__":
    ...