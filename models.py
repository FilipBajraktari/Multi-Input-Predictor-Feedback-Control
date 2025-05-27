import h5py
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from utils import preprocess_control_inputs

    
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

    def forward(self, state, controls):
        """
        Inputs:
            - state:   (batch_size, n)
            - control: (batch_size, m, num_points)
        """
        combined = torch.cat(
            (state.unsqueeze(-1).expand(-1, -1, self.num_points), controls),
            dim=1,
        )
        return self.fno(combined)[..., -1]
    
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
            X = torch.tensor(sample['X'][:], dtype=torch.float32)
            U = preprocess_control_inputs([
                torch.tensor(sample[f'U{i}'][:], dtype=torch.float32)
                for i in range(self.m_inputs)
            ])
            P = torch.tensor(sample['P'][:], dtype=torch.float32)

        return X, U, P
    
def get_data_loaders(dataset: Dataset):
    # Define split sizes
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10,
                       help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='input batch size for training')
    args = parser.parse_args()

    dataset = PredictorOperatorDataset("data/const_delay.h5")
    train_loader, val_loader, test_loader = get_data_loaders(dataset)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNOProjection(
        n_states=dataset.n_states,
        m_inputs=dataset.m_inputs,
        num_points=dataset.num_points,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        for batch_idx, (state, controls, prediction) in enumerate(train_loader):
            # Move data to device
            state = state.to(device)
            controls = controls.to(device)
            prediction = prediction.to(device)

            # Forward pass
            outputs = model(state, controls)
            loss = criterion(outputs, prediction)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print training progress
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for state, controls, prediction in val_loader:
                state = state.to(device)
                controls = controls.to(device)
                prediction = prediction.to(device)
                outputs = model(state, controls)
                val_loss += criterion(outputs, prediction).item()
        
        print(f'Epoch [{epoch+1}/{args.epochs}], Validation Loss: {val_loss/len(val_loader):.6f}')
