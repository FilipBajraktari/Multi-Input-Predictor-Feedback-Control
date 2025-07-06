from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from models import get_model_class


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
    def N(self) -> int:
        return len(self.t)
    
    @property
    def NX(self) -> int:
        return len(self.x)


@dataclass
class InferenceConfig:
    name: str
    P1: str
    P2: str

class Unicycle:
    
    def __init__(self, init_state, simulation_config: SimulationConfig, inference_config: InferenceConfig = None):

        assert all(simulation_config.delays[i-1] < simulation_config.delays[i]
            for i in range(1, len(simulation_config.delays))), \
            "It is assumed that the order D1 < D2 < ... < Dm holds!"

        # Simulation config
        self.config: SimulationConfig = simulation_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize state
        assert len(init_state) == 3
        self.n = len(init_state)
        self.X = np.copy(init_state)

        # Initial input history
        # TODO: For now it is hardcoded and assumed that init_inputs are zero
        self.m = 2
        self.control_pdes = np.zeros((self.m, self.m * simulation_config.NX), dtype=np.float32)
        self.delay_lookup = np.zeros((self.m, self.m))
        for i in range(self.m):
            delay_amount = self.config.delays[i]
            for j in range(i, -1, -1):
                next_delay_amount = self.config.delays[j-1] if j > 0 else 0
                self.delay_lookup[i, j] = delay_amount - next_delay_amount
                delay_amount = next_delay_amount

        # Exact/ML predictors
        self.P = self.predict()
        self.P_hat = None
        self.ml_predictors = None
        if inference_config is not None:
            name = inference_config.name

            # P1
            P1 = inference_config.P1
            self.P1 = get_model_class(name).load(P1, self.device)
            self.P1.eval()

            # P2
            P2 = inference_config.P2
            self.P2 = get_model_class(name).load(P2, self.device)
            self.P2.eval()

            self.ml_predictors = [self.P1, self.P2]
            self.P_hat = self.ml_predict()

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
        assert self.ml_predictors is not None
        return [
            sum(p.numel() for p in ml_predictor.parameters())
            for ml_predictor in self.ml_predictors
        ]

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
    
    def controller(self, t):
        P = self.P_hat if self.ml_predictors is not None else self.P
        return np.array([
            self.control(P[i, -1], t + self.config.delays[i])[i]
            for i in range(self.m)
        ])

    def predict(self):
        self.P = np.zeros((self.m, self.config.NX, self.n), dtype=np.float32)
        for i in range(self.m):

            # Initialization
            self.P[i, 0] = self.P[i-1, -1] if i>0 else self.X

            # Calculate
            delays = self.config.delays
            varphi = (delays[i] - delays[i-1]) if i > 0 else delays[0]
            integral_tmp = np.zeros((self.config.NX, self.n), dtype=np.float32)
            for j in range(1, self.config.NX):
                integral_tmp[j] = Unicycle.dynamics(self.P[i, j-1], self.control_pdes[:, i*self.config.NX + j-1])
                self.P[i, j] = self.P[i, 0] + varphi*np.array([np.trapezoid(integral_tmp[0:j+1, 0], dx=self.config.dx),
                                                    np.trapezoid(integral_tmp[0:j+1, 1], dx=self.config.dx),
                                                    np.trapezoid(integral_tmp[0:j+1, 2], dx=self.config.dx)])
        
        return self.P
    
    def ml_predict(self):
        assert self.ml_predictors is not None

        NX = self.config.NX
        delays = self.config.delays

        self.P_hat = np.zeros((self.m, self.config.NX, self.n), dtype=np.float32)
        for i, ml_predictor in enumerate(self.ml_predictors):
            Q = self.P_hat[i-1, -1] if i > 0 else self.X
            X = torch.tensor(Q, device=self.device).unsqueeze(0)
            start = i * NX
            end = (i+1) * NX
            U = torch.tensor(self.control_pdes[:, start:end], device=self.device).unsqueeze(0)
            varphi = torch.tensor(
                [(delays[i] - delays[i-1]) if i > 0 else delays[0]],
                device=self.device,
            ).unsqueeze(0)

            self.P_hat[i] = ml_predictor(X, U, varphi).squeeze(0).cpu().detach().numpy()

        return self.P_hat

    def step(self, U):

        # Update system state
        self.X += self.dynamics(self.X, self.control_pdes[:, 0]) * self.config.dt

        # Update control history
        dt = self.config.dt
        dx = self.config.dx
        NX = self.config.NX
        for i in range(self.m):
            for j in range(i, -1, -1):
                varphi = self.delay_lookup[i, j]
                start = j * NX
                end = (j+1) * NX
                self.control_pdes[i, start:end-1] += dt/(varphi*dx) * (self.control_pdes[i, start+1:end] - self.control_pdes[i, start:end-1])
                self.control_pdes[i, end-1] = self.control_pdes[i, end] if j < i else U[i]
        
        # Update system predictors
        if self.predict_exact:
            self.P = self.predict()
        if self.predict_ml and self.ml_predictors is not None:
            self.P_hat = self.ml_predict()

        return self.X, self.P, self.P_hat, self.control_pdes


def simulate_system(uni: Unicycle):
    t = uni.config.t
    N = uni.config.N

    states = np.zeros((N, 3), dtype=np.float32)
    controls = np.zeros((N, 2), dtype=np.float32)
    P = np.zeros((N,) + uni.P.shape, dtype=np.float32)
    P_hat = np.zeros((N,) + uni.P.shape, dtype=np.float32)
    control_pdes = np.zeros((N,) + uni.control_pdes.shape, dtype=np.float32)

    states[0] = uni.X
    P[0] = uni.P
    P_hat[0] = uni.P_hat
    for i in range(1, N):
        controls[i-1] = uni.controller(t[i-1])
        states[i], P[i], P_hat[i], control_pdes[i] = uni.step(controls[i-1])
    controls[N-1] = uni.controller(t[N-1])

    return states, controls, P, P_hat, control_pdes


if __name__ == '__main__':
    ...