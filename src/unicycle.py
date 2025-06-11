import numpy as np

from system import System
from models import get_model_class


class Unicycle(System):

    def __init__(self, init_state, init_inputs, config, predictor_model_config=None):
        super().__init__(init_state, init_inputs, config)

        # Initialize ML predictor
        if predictor_model_config is not None:
            name = predictor_model_config.name
            path = predictor_model_config.path
            self.ml_predictor_model = get_model_class(name).load(path, self.device)
            self.ml_predictor_model.eval()

            self.P_hat = self.ml_predictor()

    def dynamics(self, X, U):
        # X = [x, y, theta]
        # U = [omega, v]

        assert (
            isinstance(X, np.ndarray) and (X.shape == (3,))
        ), "X must be a numpy array of 3 elements"

        assert (
            isinstance(U, np.ndarray) and (U.shape == (2,))
        ), "U must be a numpy array of 2 elements"

        X[0] += U[1] * np.cos(X[2]) * self.config.dt
        X[1] += U[1] * np.sin(X[2]) * self.config.dt
        X[2] += U[0] * self.config.dt

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


if __name__ == '__main__':
    ...