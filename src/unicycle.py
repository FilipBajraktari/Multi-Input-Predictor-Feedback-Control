from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from tqdm import tqdm


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


class Unicycle:
    
    def __init__(self, init_state, config: SimulationConfig):

        assert all(config.delays[i-1] < config.delays[i]
            for i in range(1, len(config.delays))), \
            "It is assumed that the order D1 < D2 < ... < Dm holds!"

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
        self.control_pdes = np.zeros((self.m, self.m * config.NX), dtype=np.float32)
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
        P = self.P if self.ml_predictor_model is None else self.P_hat
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
        if self.predict_ml and self.ml_predictor_model is not None:
            self.P_hat = None

        return self.X, self.P, self.P_hat


def simulate_system(uni: Unicycle):
    t = uni.config.t
    N = uni.config.N

    states = np.zeros((N, 3), dtype=np.float32)
    controls = np.zeros((N, 2), dtype=np.float32)
    P = np.zeros((N,) + uni.P.shape, dtype=np.float32)
    P_hat = np.zeros((N,) + uni.P.shape, dtype=np.float32)

    states[0] = uni.X
    P[0] = uni.P
    P_hat[0] = uni.P_hat
    for i in tqdm(range(1, N)):
        controls[i-1] = uni.controller(t[i-1])
        states[i], P[i], P_hat[i] = uni.step(controls[i-1])
    controls[N-1] = uni.controller(t[N-1])

    return states, controls, P, P_hat


if __name__ == '__main__':
    ...