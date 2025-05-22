import matplotlib.pyplot as plt
import numpy as np

from unicycle import Unicycle


# Simulation Params
dt = 1e-4
T = 20
t = np.arange(0, T + dt, dt)
N = t.shape[0]
D = 1
ND = int(D / dt)

def simulate_system(sys: Unicycle):
    states = np.zeros((N, 3))
    controls = np.zeros((N-1, 2))

    states[0] = sys.X
    for i in range(1, N):
        controls[i-1] = sys.controller(t[i-1])
        states[i], _ = sys.step(controls[i-1])
    return states, controls

def main():

    # Simulation
    X0 = np.array([1, 1, 1])
    init_input = np.zeros((ND, 2))
    sys = Unicycle(X0, init_input, dt)
    states, controls = simulate_system(sys)

    # Plot Graphics
    plot_size = [6.4, 4.8]
    ni, nj = 1, 3
    fig, axs = plt.subplots(ni, nj, figsize=(nj * plot_size[0], ni * plot_size[1]), squeeze=False)

    # X coordinate
    axs[0, 0].plot(t, states[:, 0])
    axs[0, 0].set_xlabel("t")
    axs[0, 0].set_ylabel("X(t)")
    axs[0, 0].set_title("X coordinate")
    axs[0, 0].grid(True)

    # Y coordinate
    axs[0, 1].plot(t, states[:, 1])
    axs[0, 1].set_xlabel("t")
    axs[0, 1].set_ylabel("Y(t)")
    axs[0, 1].set_title("Y coordinate")
    axs[0, 1].grid(True)

    # Theta
    axs[0, 2].plot(t, states[:, 2])
    axs[0, 2].set_xlabel("t")
    axs[0, 2].set_ylabel("theta(t)")
    axs[0, 2].set_title("theta")
    axs[0, 2].grid(True)

    plt.show()

if __name__ == "__main__":
    main()