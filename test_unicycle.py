import matplotlib.pyplot as plt
import numpy as np

from unicycle import Unicycle


# Simulation Params
dt = 1e-4
T = 20
t = np.arange(0, T + dt, dt)
N = t.shape[0]
D1, D2 = 0.25, 1
ND1, ND2 = int(D1 / dt), int(D2 / dt)

def simulate_system(sys: Unicycle):
    states = np.zeros((N, 3))
    controls = np.zeros((N-1, 2))

    states[0] = sys.X
    for i in range(1, N):
        controls[i-1] = sys.controller(t[i-1])
        states[i], _ = sys.step(controls[i-1], t[i-1])
    return states, controls

def main():

    # Simulation
    X0 = np.array([1, 1, 1])
    init_inputs = (np.zeros(ND1), np.zeros(ND2))
    sys = Unicycle(X0, init_inputs, dt)
    states, controls = simulate_system(sys)

    # Plot Graphics
    fig = plt.figure(figsize=(10, 6))
    gs =  fig.add_gridspec(2, 6)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])

    # X coordinate
    ax1.plot(t, states[:, 0])
    ax1.set_xlabel("t")
    ax1.set_ylabel("X(t)")
    ax1.grid(True)

    # Y coordinate
    ax2.plot(t, states[:, 1])
    ax2.set_xlabel("t")
    ax2.set_ylabel("Y(t)")
    ax2.grid(True)

    # Theta
    ax3.plot(t, states[:, 2])
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\theta(t)$")
    ax3.grid(True)

    # Anuglar velocity - U1
    ax4.plot(t[:-1], controls[:, 0])
    ax4.set_xlabel("t")
    ax4.set_ylabel(r"$\omega(t)$")
    ax4.grid(True)

    # Linear velocity - U2
    ax5.plot(t[:-1], controls[:, 1])
    ax5.set_xlabel("t")
    ax5.set_ylabel(r"$\nu(t)$")
    ax5.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()