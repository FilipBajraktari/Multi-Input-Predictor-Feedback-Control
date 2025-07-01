from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from models import preprocess_control_inputs


@dataclass
class SimulationConfig:
    T: float
    dt: float
    dx: float
    delays: List[float]

    @property
    def t(self) -> np.ndarray:
        return np.arange(0, self.T + self.dt, self.dt)
    
    @property
    def x(self) -> np.ndarray:
        return np.arange(0, 1 + self.dx, self.dx)
    
    @property
    def NDs(self) -> List[int]:
        return [int(delay / self.dt) for delay in self.delays]
    
    @property
    def N(self) -> int:
        return len(self.t)
    
    @property
    def NX(self) -> int:
        return len(self.x)


class System(ABC):

    def __init__(self, init_state, init_inputs, simulation_cfg):

        # Simulation config
        self.config: SimulationConfig = simulation_cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize state
        self.n = len(init_state)
        self.X = np.copy(init_state)

        # Initial input history
        self.m = len(init_inputs)
        self.NDs = np.zeros(self.m, dtype=np.uint32)
        self.delays = np.zeros(self.m, dtype=np.float32)
        self.control_pdes = []
        for i, init_input in enumerate(init_inputs):
            self.NDs[i] = init_input.shape[0]
            self.delays[i] = self.NDs[i] * self.config.dt
            pde_sol = np.append(init_input, 0).astype(np.float32)
            self.control_pdes.append(pde_sol)

        assert all(self.control_pdes[i-1].shape[0] <= self.control_pdes[i].shape[0]
            for i in range(1, len(self.control_pdes))), \
            "It is assumed that the order D1 <= D2 <= ... <= Dm holds!"

        # Exact/ML predictors
        self.P = self.predictor(t0=0)
        self.P_hat = None
        self.ml_predictor_model = None

        # These variables are used for perf estimates to manually swith them on/off
        self.predict_exact = True
        self.predict_ml = True

    def use_only_exact_predictor(self):
        self.predict_exact = True
        self.predict_ml = False

    def use_only_ml_predictor(self):
        self.predict_exact = False
        self.predict_ml = True

    def use_both_predictors(self):
        self.predict_exact = True
        self.predict_ml = True

    def get_model_params(self):
        assert self.ml_predictor_model is not None
        return sum(p.numel() for p in self.ml_predictor_model.parameters())

    @abstractmethod
    def dynamics(X, U):
        ...
    
    @abstractmethod
    def control(X, t):
        ...

    def controller(self, t):
        P = self.P if self.ml_predictor_model is None else self.P_hat
        return np.array([
            self.control(P[:, self.NDs[i]-1], t + self.delays[i])[i]
            for i in range(self.m)
        ])
    
    def predictor(self, t0):
        P = np.repeat(np.expand_dims(self.X, axis=1), self.NDs[-1]+1, axis=1)
        for i in range(self.m):

            lower_bound = 0 if i == 0 else self.control_pdes[i-1].shape[0]-1
            for j in range(lower_bound, self.control_pdes[i].shape[0]-1):
                
                # Construct the input signal
                U = np.array([
                    self.control(P[:, j], t0 + j * self.config.dt)[k]
                    if control_pde.shape[0] - 1 <= j
                    else control_pde[j]
                    for k, control_pde in enumerate(self.control_pdes)
                ])

                # Propagate predictor
                P[:, j+1] = self.dynamics(P[:, j], U)

        return P[:, 1:]
    
    def ml_predictor(self):
        assert self.ml_predictor_model is not None

        X = torch.tensor(self.X, device=self.device).unsqueeze(0)
        U = preprocess_control_inputs([
            torch.tensor(control_pde[:-1], device=self.device)
            for control_pde in self.control_pdes
        ]).unsqueeze(0)
        P = self.ml_predictor_model(X, U).squeeze(0).cpu().detach().numpy()

        return P
    
    def step(self, U, t):
        
        # Update system state
        self.X = self.dynamics(
            self.X,
            np.array([control_pde[0] for control_pde in self.control_pdes]),
        )

        # Update control inputs
        for i in range(self.m):
            self.control_pdes[i][-1] = U[i]
            self.control_pdes[i][:-1] = self.control_pdes[i][1:]

        # Update system predictors
        if self.predict_exact:
            self.P = self.predictor(t)
        if self.predict_ml and self.ml_predictor_model is not None:
            self.P_hat = self.ml_predictor()

        return self.X, self.P, self.P_hat
    

class DimensionlessSystem(ABC):
    
    def __init__(self, init_state, config: SimulationConfig):

        assert all(config.delays[i-1] <= config.delays[i]
            for i in range(1, len(config.delays))), \
            "It is assumed that the order D1 <= D2 <= ... <= Dm holds!"

        # Simulation config
        self.config: SimulationConfig = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize state
        assert len(init_state) == 3
        self.n = len(init_state)
        self.X = np.copy(init_state)

        # Initial input history
        # TODO: For now it is hardcoded and assumed that init_inputs are zero
        self.m = 2
        self.control_pdes = np.zeros((self.m, self.m * config.NX))

        # TEST
        self.control_pdes[0] = 1

    @staticmethod
    def dynamics(X, U):
        f1 = U[1] * np.cos(X[2])
        f2 = U[1] * np.sin(X[2])
        f3 = U[0]
        return np.array([f1, f2, f3])
    
    @staticmethod
    def control(X, t):
        P = X[0] * np.cos(X[2]) + X[1] * np.sin(X[2])
        Q = X[0] * np.sin(X[2]) - X[1] * np.cos(X[2])
        U1 = -5 * P**2 * np.cos(3*t) - P * Q * (1 + 25 * np.cos(3*t)**2) - X[2]
        U2 = -P + 5 * Q * (np.sin(3*t) - np.cos(3*t)) + Q * U1
        return np.array([U1, U2])

    def predict(self):
        P = np.zeros((self.m, self.config.NX, self.n))
        for i in range(self.m):

            # Initialization
            P[i, 0] = P[i-1, -1] if i>0 else self.X

            # Calculate
            delays = self.config.delays
            varphi = (delays[i] - delays[i-1]) if i > 0 else delays[0]
            integral_tmp = np.zeros((self.config.NX, self.n))
            for j in range(1, self.config.NX):
                integral_tmp[j] = DimensionlessSystem.dynamics(P[i, j-1], self.control_pdes[:, i*self.config.NX + j-1])
                P[i, j] = P[i, 0] + varphi*np.array([np.trapezoid(integral_tmp[0:j+1, 0], dx=self.config.dx),
                                                    np.trapezoid(integral_tmp[0:j+1, 1], dx=self.config.dx),
                                                    np.trapezoid(integral_tmp[0:j+1, 2], dx=self.config.dx)])
                
        return P[:, -1]

    def step(self, U):

        # Update system state
        self.X = self.dynamics(self.X, self.control_pdes[:, 0])

        # Update control history
        dt = self.config.dt
        dx = self.config.dx
        delays = self.config.delays
        NX = self.config.NX
        for i in range(self.m):
            for j in range(i-1, -1, -1):
                varphi = (delays[i] - delays[i-1]) if i > 0 else delays[0]
                self.control_pdes[i, 0:NX-1] += dt/(varphi*dx) * (self.control_pdes[i, 1:NX] - self.control_pdes[i, 0:NX-1])
                self.control_pdes[i][:-1] = self.control_pdes[i+1][1:] if i < j-1 else U[i]

        return self.X, self.P, self.P_hat


def simulate_system(system: System):
    t = system.config.t
    NDs = system.config.NDs
    N = system.config.N

    states = np.zeros((N, 3), dtype=np.float32)
    controls = np.zeros((N, 2), dtype=np.float32)
    P = np.zeros((N, 3, NDs[-1]), dtype=np.float32)
    P_hat = np.zeros((N, 3, NDs[-1]), dtype=np.float32)

    states[0] = system.X
    P[0] = system.P
    P_hat[0] = system.P_hat
    for i in range(1, N):
        controls[i-1] = system.controller(t[i-1])
        states[i], P[i], P_hat[i] = system.step(controls[i-1], t[i-1])
    controls[N-1] = system.controller(t[N-1])

    return states, controls, P, P_hat


if __name__ == '__main__':
    init_state = np.ones(3, dtype=np.float32)
    config = SimulationConfig(T=10,dt=0.001,dx=0.01,delays=[0.25,0.60])
    uni = DimensionlessSystem(init_state, config)
    print(uni.predict())