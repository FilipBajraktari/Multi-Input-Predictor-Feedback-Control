from abc import ABC, abstractmethod

import numpy as np


class System(ABC):

    def __init__(self, init_state, init_inputs, dt):

        # Initialize state
        self.n = len(init_state)
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

    @abstractmethod
    def dynamics(X, U, dt):
        pass
    
    @abstractmethod
    def control(X, t):
        pass

    @abstractmethod
    def controller(self, t):
        pass
    
    def predictor(self, t0=0):
        self.P = np.repeat(np.expand_dims(self.X, axis=0), self.m, axis=0)
        for i in range(self.m):

            lower_bound = 0 if i == 0 else self.control_pdes[i-1].shape[0] - 1
            for j in range(lower_bound, self.control_pdes[i].shape[0] - 1):
                
                # Construct the input signal
                U = np.array([
                    self.control(self.P[-1], t0 + j * self.dt)[k]
                    if control_pde.shape[0] - 1 <= j
                    else control_pde[j]
                    for k, control_pde in enumerate(self.control_pdes)
                ])

                # Propagate predictors
                P = self.dynamics(self.P[-1], U, self.dt)
                for k in range(self.m):
                    if j < self.control_pdes[k].shape[0] - 1:
                        self.P[k] = P

        return self.P
    
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
                self.control(self.P[i], t + self.delays[i])[k]
                if k < i
                else control_pde[nx_i-1]
                for k, control_pde in enumerate(self.control_pdes)
            ])

            self.P[i] = self.dynamics(self.P[i], U, self.dt)

        for i in range(self.m):
            self.control_pdes[i][0:-1] = self.control_pdes[i][1:]

        return self.X, self.P
