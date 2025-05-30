import numpy as np

from system import System
from models import FNOProjection


class Unicycle(System):

    def __init__(self, init_state, init_inputs, dt, ml_predictor_path=None):
        super().__init__(init_state, init_inputs, dt)

        # Initialize ML predictor
        if ml_predictor_path is not None:
            self.ml_predictor_model = FNOProjection.load(ml_predictor_path, self.device)
            self.ml_predictor_model.eval()

        # Initialize predicotr variables
        self.P = self.exact_predictor()
        if self.ml_predictor_model is not None:
            self.P_hat = self.ml_predictor()

    def dynamics(self, X, U, dt):
        # X = [x, y, theta]
        # U = [omega, v]

        assert (
            isinstance(X, np.ndarray) and (X.shape == (3,))
        ), "X must be a numpy array of 3 elements"

        assert (
            isinstance(U, np.ndarray) and (U.shape == (2,))
        ), "U must be a numpy array of 2 elements"

        X[0] += U[1] * np.cos(X[2]) * dt
        X[1] += U[1] * np.sin(X[2]) * dt
        X[2] += U[0] * dt

        return X
    
    def control(self, X, t):
        # X = [x, y, theta]

        assert (
            isinstance(X, np.ndarray) and (X.shape == (3,))
        ), "X must be a numpy array of 3 elements"

        P = X[0] * np.cos(X[2]) + X[1] * np.sin(X[2])
        Q = X[0] * np.sin(X[2]) - X[1] * np.cos(X[2])
        U1 = -5 * P**2 * np.cos(3*t) - P * Q * (1 + 25 * np.cos(3*t)**2) - X[2]
        U2 = -P + 5 * Q * (np.sin(3*t) - np.cos(3*t)) + Q * U1

        # [angular velocity, linear velocity]
        return np.array([U1, U2])