from dataclasses import dataclass
from pathlib import Path
import tomli
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

import wandb
from models import ModelConfig, PredictorOperatorDataset, get_data_loaders, get_model_class


@dataclass
class ModelTrainingConfig:
    model: ModelConfig
    epochs: int
    lr: float
    weight_decay: float
    gamma: float
    batch_size: int
    dataset: str


def train(config: ModelTrainingConfig):

    # Load dataset
    abs_dataset_path = (Path(__file__).parent.parent / f'data/{config.dataset}.h5').resolve()
    dataset = PredictorOperatorDataset(abs_dataset_path)
    train_loader, val_loader, test_loader = get_data_loaders(dataset, config.batch_size)

    # Instantiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model_class(config.model.name)(
        n_states=dataset.n_states,
        m_inputs=dataset.m_inputs,
        num_points=dataset.num_points,
        dt=dataset.dt,
        delays=dataset.delays,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=config.gamma)
    criterion = nn.MSELoss()

    # Logging
    wandb.init(
        project="unicycle",
        config={
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.lr,
            "model_architecture": str(model),
            "dataset": config.dataset,
        }
    )
    wandb.watch(model, log="all", log_freq=100)
    
    # Model training
    for epoch in range(config.epochs):

        model.train()
        epoch_loss = 0.0
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

            # Accumulate epoch loss
            epoch_loss += loss.item()
            
            # Print training progress
            if (batch_idx+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.8f}')
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch-level training loss
        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({
            "epoch_train_loss": avg_train_loss,
        })

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
        
        avg_val_loss = val_loss / len(val_loader)
    
        # Log validation metrics
        wandb.log({
            "epoch_val_loss": avg_val_loss,
        })
        print(f'Epoch [{epoch+1}/{config.epochs}], Validation Loss: {avg_val_loss:.8f}')

    abs_model_path = (Path(__file__).parent.parent / f'models/{config.model.path}.pth').resolve()
    model.save(abs_model_path)
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config.toml',
                       help='relative path to config file from project directory')
    args = parser.parse_args()

    abs_cofig_path = (Path(__file__).parent.parent / args.config).resolve()
    with open(abs_cofig_path, 'rb') as f:

        # Extract config data from .toml file
        data = tomli.load(f)
        data['training']['model'] = ModelConfig(**data['training']['model'])
        config = ModelTrainingConfig(**data['training'])

        train(config)