import numpy as np

from system import System


class Unicycle(System):

    def dynamics(self, X, U, dt):
        # X = [x, y, theta]
        # U = [v, omega]

        assert (
            isinstance(X, np.ndarray) 
            and (X.shape == (3,)) 
            and (X.dtype == np.float64)
        ), "X must be a numpy array of 3 elements with dtype float64"

        assert (
            isinstance(U, np.ndarray) 
            and (U.shape == (2,)) 
            and (U.dtype == np.float64)
        ), "U must be a numpy array of 2 elements with dtype float64"

        X[0] += U[1] * np.cos(X[2]) * dt
        X[1] += U[1] * np.sin(X[2]) * dt
        X[2] += U[0] * dt

        return X
    
    def control(self, X, t):
        # X = [x, y, theta]

        assert (
            isinstance(X, np.ndarray) 
            and (X.shape == (3,)) 
            and (X.dtype == np.float64)
        ), "X must be a numpy array of 3 elements with dtype float64"

        P = X[0] * np.cos(X[2]) + X[1] * np.sin(X[2])
        Q = X[0] * np.sin(X[2]) - X[1] * np.cos(X[2])
        U1 = -5 * P**2 * np.cos(3*t) - P * Q * (1 + 25 * np.cos(3*t)**2) - X[2]
        U2 = -P + 5 * Q * (np.sin(3*t) - np.cos(3*t)) + Q * U1

        # [angular velocity, linear velocity]
        return np.array([U1, U2])
    
    def controller(self, t):
        return np.array([self.control(self.P[i], t + self.delays[i])[i] for i in range(self.m)])