import numpy as np

class Unicycle:

    def __init__(self, init_state, init_input, dt):
        # Initialize state
        self.X = init_state.astype(np.float64)

        # nx = [D/dt] + 1
        self.pde_sol = np.vstack((init_input, np.zeros(2))).astype(np.float64)
        self.nx = self.pde_sol.shape[0]
        self.dt = dt

        # Initialize predictor state
        self.P = self.predictor()

    @staticmethod
    def dynamics(X, U, dt):
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

        X[0] += U[0] * np.cos(X[2]) * dt
        X[1] += U[0] * np.sin(X[2]) * dt
        X[2] += U[1] * dt

        return X
    
    @staticmethod
    def control(X, t):
        # X = [x, y, theta]

        assert (
            isinstance(X, np.ndarray) 
            and (X.shape == (3,)) 
            and (X.dtype == np.float64)
        ), "X must be a numpy array of 3 elements with dtype float64"

        P = X[0] * np.cos(X[2]) + X[1] * np.sin(X[2])
        Q = X[0] * np.sin(X[2]) - X[1] * np.cos(X[2])
        omega = -5 * P**2 * np.cos(3*t) - P * Q * (1 + 25 * np.cos(3*t)**2) - X[2]
        nu = -P + 5 * Q * (np.sin(3*t) - np.cos(3*t)) + Q * omega

        return np.array([nu, omega])
    
    def predictor(self):
        self.P = np.copy(self.X)
        for i in range(self.nx-1):
            self.P = Unicycle.dynamics(self.P, self.pde_sol[i], self.dt)
        return self.P
    
    def controller(self, t):
        # This is a placeholder for future where insted of self.P we have ML predictor
        P = self.P
        return Unicycle.control(P, t)
    
    def step(self, U):

        self.pde_sol[-1] = U

        self.X = Unicycle.dynamics(self.X, self.pde_sol[0], self.dt)
        self.P = Unicycle.dynamics(self.P, self.pde_sol[-1], self.dt)

        self.pde_sol[0:-1] = self.pde_sol[1:]

        return self.X, self.P