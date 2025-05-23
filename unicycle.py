import numpy as np

class Unicycle:

    def __init__(self, init_state, init_inputs, dt):

        # Initialize state
        self.X = init_state.astype(np.float64)

        # Initial input history
        self.m = len(init_inputs)
        self.control_pdes = []
        self.delays = []
        for init_input in init_inputs:
            pde_sol = np.append(init_input, 0).astype(np.float64)
            self.control_pdes.append(pde_sol)
            self.delays.append(init_input.shape[0] * dt)

        assert all(self.control_pdes[i-1].shape[0] <= self.control_pdes[i].shape[0]
            for i in range(1, len(self.control_pdes))), \
            "It is assumed that the order D1 < D2 < ... < Dm holds!"

        # Simulation time step
        self.dt = dt

        # Initialize predictor state
        self.P = self.predictor()
        print(self.P)

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

        X[0] += U[1] * np.cos(X[2]) * dt
        X[1] += U[1] * np.sin(X[2]) * dt
        X[2] += U[0] * dt

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
        U1 = -5 * P**2 * np.cos(3*t) - P * Q * (1 + 25 * np.cos(3*t)**2) - X[2]
        U2 = -P + 5 * Q * (np.sin(3*t) - np.cos(3*t)) + Q * U1

        # [angular velocity, linear velocity]
        return np.array([U1, U2])
    
    def predictor(self, t0=0):
        self.P = np.repeat(np.expand_dims(self.X, axis=0), self.m, axis=0)
        for i in range(self.m):

            lower_bound = 0 if i == 0 else self.control_pdes[i-1].shape[0] - 1
            for j in range(lower_bound, self.control_pdes[i].shape[0] - 1):
                
                # Construct the input signal
                U = np.array([
                    Unicycle.control(self.P[-1], t0 + j * self.dt)[k]
                    if control_pde.shape[0] - 1 <= j
                    else control_pde[j]
                    for k, control_pde in enumerate(self.control_pdes)
                ])

                # Propagate predictors
                P = Unicycle.dynamics(self.P[-1], U, self.dt)
                for k in range(self.m):
                    if j < self.control_pdes[k].shape[0] - 1:
                        self.P[k] = P

        return self.P
    
    def controller(self, t):
        return np.array([Unicycle.control(self.P[i], t + self.delays[i])[i] for i in range(self.m)])
    
    def step(self, U, t):

        for i in range(self.m):
            self.control_pdes[i][-1] = U[i]
        
        self.X = Unicycle.dynamics(
            self.X,
            np.array([control_pde[0] for control_pde in self.control_pdes]),
            self.dt,
        )
        for i in range(self.m):
            nx_i = self.control_pdes[i].shape[0]

            # Construct the input signal
            U = np.array([
                Unicycle.control(self.P[i], t + self.delays[i])[k]
                if k < i
                else control_pde[nx_i-1]
                for k, control_pde in enumerate(self.control_pdes)
            ])

            self.P[i] = Unicycle.dynamics(self.P[i], U, self.dt)

        for i in range(self.m):
            self.control_pdes[i][0:-1] = self.control_pdes[i][1:]

        return self.X, self.P