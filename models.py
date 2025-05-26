import torch
import torch.nn as nn
from neuralop.models import FNO


class NeuralPredictor(torch.nn.Module):
    def __init__(self, n_input: int, n_state: int, seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_state = n_state
        self.seq_len = seq_len

        self.n_input_channel = n_input + n_state
        self.n_output_channel = n_state
        self.mse_loss = torch.nn.MSELoss()
    
class FNOProjection(NeuralPredictor):
    def __init__(self, n=3, m=2, D_max=1.0, width=64, modes=16):
        super().__init__()
        self.n = n  # Dimension of ℝⁿ
        self.m = m  # Number of delay channels
        self.D_max = D_max
        
        # Projection for ℝⁿ component
        self.proj_rn = nn.Linear(n, width)
        
        # FNO for each delay channel
        self.delay_fnos = nn.ModuleList([
            FNO(n_modes=(modes,),       # 1D FNO
                hidden_channels=width,
                in_channels=1,          # Each delay is scalar-valued
                out_channels=width,
                n_layers=2)
            for _ in range(m)
        ])
        
        # Combined processing
        self.combine = FNO(
            n_modes=(modes,),
            hidden_channels=width*(m+1),
            in_channels=width*(m+1),
            out_channels=n,
            n_layers=3,
        )

    def forward(self, rn_input, *delay_inputs):
        """
        Inputs:
        - rn_input: (batch_size, n)
        - delay_inputs: m tensors of shape (batch_size, 1, num_points)
        """
        # Process ℝⁿ component
        h_rn = self.proj_rn(rn_input).unsqueeze(-1)  # (batch_size, width, 1)
        
        # Process each delay channel
        h_delays = [fno(d) for fno, d in zip(self.delay_fnos, delay_inputs)]
        
        # Combine features
        combined = torch.cat([h_rn] + h_delays, dim=1)  # (batch_size, width*(m+1), L)
        
        # Final mapping
        return self.combine(combined)
    