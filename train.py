import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

import wandb
from models import FNOProjection, PredictorOperatorDataset, get_data_loaders


def train(
        epochs: int,
        lr: float,
        batch_size: int,
        data_path: str
):
    dataset = PredictorOperatorDataset(data_path)
    train_loader, val_loader, test_loader = get_data_loaders(dataset, batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FNOProjection(
        n_states=dataset.n_states,
        m_inputs=dataset.m_inputs,
        num_points=dataset.num_points,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.00001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    criterion = nn.MSELoss()

    # Logging
    wandb.init(
        project="unicycle",
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "model_architecture": str(model),
        }
    )
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(epochs):

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
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')
        
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
        print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.6f}')

    model.save(f"models/const_distinct_delays_{epochs}_epochs.pth")
    wandb.finish()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=15,
                       help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='learning rate')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='input batch size for training')
    parser.add_argument('--data_path', type=str, default="data/const_distinct_delays.h5",
                       help='path to dataset')
    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size, args.data_path)