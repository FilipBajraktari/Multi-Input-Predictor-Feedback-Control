from abc import ABC, abstractmethod

import numpy as np
import torch

from utils import preprocess_control_inputs


class System(ABC):

    def __init__(self, init_state, init_inputs, dt):

        # Initialize state
        self.n = len(init_state)
        self.X = init_state

        # Initial input history
        self.m = len(init_inputs)
        self.NDs = np.zeros(self.m, dtype=np.uint32)
        self.delays = np.zeros(self.m, dtype=np.float32)
        self.control_pdes = []
        for i, init_input in enumerate(init_inputs):
            self.NDs[i] = init_input.shape[0]
            self.delays[i] = self.NDs[i] * dt
            pde_sol = np.append(init_input, 0).astype(np.float32)
            self.control_pdes.append(pde_sol)

        assert all(self.control_pdes[i-1].shape[0] <= self.control_pdes[i].shape[0]
            for i in range(1, len(self.control_pdes))), \
            "It is assumed that the order D1 <= D2 <= ... <= Dm holds!"

        # Simulation time step
        self.dt = dt

        # Exact/ML predictors
        self.P = None
        self.P_hat = None
        self.ml_predictor_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def dynamics(X, U, dt):
        ...
    
    @abstractmethod
    def control(X, t):
        ...

    def controller(self, t):
        if self.ml_predictor_model is None:
            return np.array([
                self.control(self.P[:, i], t + self.delays[i])[i]
                for i in range(self.m)
            ])

        self.P_hat = self.ml_predictor()
        return np.array([
            self.control(self.P_hat[:, i], t + self.delays[i])[i]
            for i in range(self.m)
        ])
    
    def exact_predictor(self, t0=0):
        P = np.repeat(np.expand_dims(self.X, axis=1), self.m, axis=1)
        for i in range(self.m):

            lower_bound = 0 if i == 0 else self.control_pdes[i-1].shape[0] - 1
            for j in range(lower_bound, self.control_pdes[i].shape[0] - 1):
                
                # Construct the input signal
                U = np.array([
                    self.control(P[:, -1], t0 + j * self.dt)[k]
                    if control_pde.shape[0] - 1 <= j
                    else control_pde[j]
                    for k, control_pde in enumerate(self.control_pdes)
                ])

                # Propagate predictors
                P_next = self.dynamics(P[:, -1], U, self.dt)
                for k in range(self.m):
                    if j < self.control_pdes[k].shape[0] - 1:
                        P[:, k] = P_next

        return P
    
    def ml_predictor(self):
        assert self.ml_predictor_model is not None

        X = torch.tensor(self.X, device=self.device).unsqueeze(0)
        U = preprocess_control_inputs([
            torch.tensor(control_pde[:-1], device=self.device)
            for control_pde in self.control_pdes
        ]).unsqueeze(0)
        P = self.ml_predictor_model(X, U).squeeze(0).cpu().detach().numpy()

        return P[:, self.NDs-1]
    
    def step(self, U, t):

        for i in range(self.m):
            self.control_pdes[i][-1] = U[i]
        
        self.X = self.dynamics(
            self.X,
            np.array([control_pde[0] for control_pde in self.control_pdes]),
            self.dt,
        )
        for i in range(self.m):
            nx_i = self.control_pdes[i].shape[0]

            # Construct the input signal
            U = np.array([
                self.control(self.P[:, i], t + self.delays[i])[k]
                if k < i
                else control_pde[nx_i-1]
                for k, control_pde in enumerate(self.control_pdes)
            ])

            self.P[:, i] = self.dynamics(self.P[:, i], U, self.dt)

        for i in range(self.m):
            self.control_pdes[i][0:-1] = self.control_pdes[i][1:]

        return self.X, self.P, self.P_hat
